from dataclasses import dataclass, field

import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg, types
from mjlab.utils.logging import print_info

from cat_mjlab.managers import ConstraintManager, ConstraintTermCfg


@dataclass(kw_only=True)
class CaTManagerBasedRlEnvCfg(ManagerBasedRlEnvCfg):
  constraints: dict[str, ConstraintTermCfg] = field(default_factory=dict)
  """Observation groups configuration. Each group (e.g., "actor", "critic") contains
  observation terms that are concatenated. Groups can have different settings for
  noise, history, and delay."""


class CaTManagerBasedRlEnv(ManagerBasedRlEnv):
  def __init__(
    self,
    cfg: CaTManagerBasedRlEnvCfg,
    device: str,
    render_mode: str | None = None,
    **kwargs,
  ) -> None:
    super().__init__(cfg, device, render_mode, **kwargs)

    self.dones = torch.zeros(self.num_envs, device=self.device)

  def load_managers(self) -> None:
    super().load_managers()
    # prepare the managers
    # -- constraint manager

    if hasattr(self.cfg, "constraints"):
      self.constraint_manager = ConstraintManager(self.cfg.constraints, self)
      print_info(f"[INFO] {self.constraint_manager}")

      self.extras["cat_terms"] = torch.zeros(self.num_envs, device=self.device)

  def step(self, action: torch.Tensor) -> types.VecEnvStepReturn:
    """Run one environment step: apply actions, simulate, compute RL signals.

    **Forward-call placement.** MuJoCo's ``mj_step`` runs forward kinematics
    *before* integration, so after stepping, derived quantities (``xpos``,
    ``xquat``, ``site_xpos``, ``cvel``, ``sensordata``) lag ``qpos``/``qvel``
    by one physics substep. Rather than calling ``sim.forward()`` twice (once
    after the decimation loop and once after the reset block), this method
    calls it **once**, right before observation computation. This single call
    refreshes derived quantities for *all* envs: non-reset envs pick up
    post-decimation kinematics, reset envs pick up post-reset kinematics.

    The tradeoff is that termination and reward managers see derived
    quantities that are stale by one physics substep (the last ``mj_step``
    ran ``mj_forward`` from *pre*-integration ``qpos``). In practice, the
    staleness is negligible for reward shaping and termination
    checks. Critically, the staleness is *consistent*: every env,
    every step, always sees the same lag, so the MDP is well-defined
    and the value function can learn the correct mapping.

    .. note::

      Event and command authors do not need to call ``sim.forward()``
      themselves. This method handles it. The only constraint is: do not
      read derived quantities (``root_link_pose_w``, ``body_link_vel_w``,
      etc.) in the same function that writes state
      (``write_root_state_to_sim``, ``write_joint_state_to_sim``, etc.).
      See :ref:`faq` for details.
    """
    self.action_manager.process_action(action.to(self.device))

    for _ in range(self.cfg.decimation):
      self._sim_step_counter += 1
      self.action_manager.apply_action()
      self.scene.write_data_to_sim()
      self.sim.step()
      self.scene.update(dt=self.physics_dt)

    # Update env counters.
    self.episode_length_buf += 1
    self.common_step_counter += 1

    # Check terminations and compute rewards.
    # NOTE: Derived quantities (xpos, xquat, ...) are stale by one physics
    # substep here. See the docstring above for why this is acceptable.
    self.reset_buf = self.termination_manager.compute()
    self.reset_terminated = self.termination_manager.terminated
    self.reset_time_outs = self.termination_manager.time_outs

    # -- CaT constraints prob computation
    if hasattr(self.cfg, "constraints"):
      cstr_prob: torch.Tensor = self.constraint_manager.compute()
      # -- constrained reward computation
      self.reward_buf = self.reward_manager.compute(dt=self.step_dt) * (1.0 - cstr_prob)
      self.dones = cstr_prob.clone()
      self.extras["cat_terms"] = cstr_prob.clone()
    else:
      self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
      self.dones = torch.zeros(self.num_envs, device=self.device)

    self.metrics_manager.compute()

    # Reset envs that terminated/timed-out and log the episode info.
    reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if len(reset_env_ids) > 0:
      self.dones[reset_env_ids] = 1.0
      self._reset_idx(reset_env_ids)
      self.scene.write_data_to_sim()

    # Single forward() call: recompute derived quantities from current
    # qpos/qvel for every env. For non-reset envs this resolves the
    # one-substep staleness left by mj_step; for reset envs it picks up
    # the freshly written reset state.
    self.sim.forward()

    self.command_manager.compute(dt=self.step_dt)

    if "step" in self.event_manager.available_modes:
      self.event_manager.apply(mode="step", dt=self.step_dt)
    if "interval" in self.event_manager.available_modes:
      self.event_manager.apply(mode="interval", dt=self.step_dt)

    self.sim.sense()
    self.obs_buf = self.observation_manager.compute(update_history=True)

    return (
      self.obs_buf,
      self.reward_buf,
      # self.dones,
      self.reset_terminated,
      self.reset_time_outs,
      self.extras,
    )

  def _reset_idx(self, env_ids: torch.Tensor | None = None) -> None:
    super()._reset_idx(env_ids)

    if hasattr(self.cfg, "constraints"):
      info = self.constraint_manager.reset(env_ids)
      self.extras["log"].update(info)
