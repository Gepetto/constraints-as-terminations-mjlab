# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION and LAAS CNRS
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.utils import resolve_callable, resolve_obs_groups
from tensordict import TensorDict

from cat_mjlab.rsl_rl.rollout_storage import CaTRolloutStorage


class CaTPPO(PPO):
  """Proximal Policy Optimization algorithm with Constraints as Termination.

  The only required changes othean than algorithmic relate to replacement of
  RolloutStorage to CaTRolloutStorage, and instantiations of CaTPPO instead of PPO.

  Reference:
      - Schulman et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
      - Chane-Sane et al. "CaT: Constraints as Terminations for Legged Locomotion Reinforcement Learning." IROS 2024.
  """

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.transition = CaTRolloutStorage.Transition()

  def compute_returns(self, obs: TensorDict) -> None:
    """Compute return and advantage targets from stored transitions."""
    st: CaTRolloutStorage = self.storage
    # Compute values for the last step
    critic_hidden_state = self.critic.get_hidden_state()
    last_values = self.critic(obs).detach()
    # Restore the critic's hidden state so the next rollout is not affected by the forward pass
    self.critic.reset(hidden_state=critic_hidden_state)
    # Compute returns and advantages
    advantage = 0
    # Modify self.gamma to a value scaled by constraints to enable CaT in this PPO implementation
    gamma_scaled = self.gamma * (1.0 - st.constraint_violations[-1, :, :])

    for step in reversed(range(st.num_transitions_per_env)):
      # If we are at the last step, bootstrap the return value
      next_values = (
        last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
      )
      # 1 if we are not in a terminal state, 0 otherwise
      next_is_not_terminal = 1.0 - st.dones[step].float()
      # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
      delta = (
        st.rewards[step]
        + next_is_not_terminal * gamma_scaled * next_values
        - st.values[step]
      )
      # Advantage: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
      advantage = delta + next_is_not_terminal * gamma_scaled * self.lam * advantage
      # Return: R_t = A(s_t, a_t) + V(s_t)
      st.returns[step] = advantage + st.values[step]
    # Compute the advantages
    st.advantages = st.returns - st.values
    # Normalize the advantages if per minibatch normalization is not used
    if not self.normalize_advantage_per_mini_batch:
      st.advantages = (st.advantages - st.advantages.mean()) / (
        st.advantages.std() + 1e-8
      )

  def process_env_step(
    self,
    obs: TensorDict,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    extras: dict[str, torch.Tensor],
  ) -> None:
    """Record one environment step and update the normalizers.
    The only required change is extraction of constraints from extras to transition object.
    """
    # Update the normalizers
    self.actor.update_normalization(obs)
    self.critic.update_normalization(obs)
    if self.rnd:
      self.rnd.update_normalization(obs)

    # Record the rewards and dones
    # Note: We clone here because later on we bootstrap the rewards based on timeouts
    self.transition.rewards = rewards.clone()
    self.transition.dones = dones

    if "cat_terms" in extras:
      self.transition.constraint_violations = extras["cat_terms"]
    else:
      self.transition.constraint_violations = torch.zeros_like(dones)

    # Compute the intrinsic rewards and add to extrinsic rewards
    if self.rnd:
      # Compute the intrinsic rewards
      self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
      # Add intrinsic rewards to extrinsic rewards
      self.transition.rewards += self.intrinsic_rewards

    # Bootstrapping on time outs
    if "time_outs" in extras:
      self.transition.rewards += self.gamma * torch.squeeze(
        self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device),  # type: ignore
        1,
      )

    # Record the transition
    self.storage.add_transition(self.transition)
    self.transition.clear()
    self.actor.reset(dones)
    self.critic.reset(dones)

  @staticmethod
  def construct_algorithm(
    obs: TensorDict, env: VecEnv, cfg: dict, device: str
  ) -> CaTPPO:
    """Construct the CaTPPO algorithm.
    The only required change compared to the base class is use of custom CaTRolloutStorage.
    """
    # Resolve class callables
    alg_class: type[CaTPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
    actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
    critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

    # Resolve observation groups
    default_sets = ["actor", "critic"]
    if "rnd_cfg" in cfg["algorithm"] and cfg["algorithm"]["rnd_cfg"] is not None:
      default_sets.append("rnd_state")
    cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

    # Resolve RND config if used
    cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)

    # Resolve symmetry config if used
    cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

    # Initialize the policy
    actor: MLPModel = actor_class(
      obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]
    ).to(device)
    print(f"Actor Model: {actor}")
    if cfg["algorithm"].pop(
      "share_cnn_encoders", None
    ):  # Share CNN encoders between actor and critic
      cfg["critic"]["cnns"] = actor.cnns  # type: ignore
    critic: MLPModel = critic_class(
      obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]
    ).to(device)
    print(f"Critic Model: {critic}")

    # Initialize the storage
    storage = CaTRolloutStorage(
      "rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device
    )

    # Initialize the algorithm
    alg: CaTPPO = alg_class(
      actor,
      critic,
      storage,
      device=device,
      **cfg["algorithm"],
      multi_gpu_cfg=cfg["multi_gpu"],
    )

    # Compile the algorithm's models if requested
    alg.compile(cfg.get("torch_compile_mode"))

    return alg
