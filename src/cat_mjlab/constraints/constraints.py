"""Common functions that can be used to define constraints for the learning environment."""

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.entity.data import EntityData
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import matrix_from_quat

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def joint_position(
  env: ManagerBasedRlEnv,
  limit: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids]) - limit


def joint_position_when_moving_forward(
  env: ManagerBasedRlEnv,
  limit: float,
  velocity_deadzone: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  cstr = (
    torch.abs(
      asset.data.joint_pos[:, asset_cfg.joint_ids]
      - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    - limit
  )
  cstr *= (torch.abs(command[:, 1]) < velocity_deadzone).float().unsqueeze(1)
  return cstr


def joint_torque(
  env: ManagerBasedRlEnv,
  limit: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids]) - limit


def joint_velocity(
  env: ManagerBasedRlEnv,
  limit: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]) - limit


def joint_acceleration(
  env: ManagerBasedRlEnv,
  limit: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.abs(asset.data.joint_acc[:, asset_cfg.joint_ids]) - limit


def upsidedown(
  env: ManagerBasedRlEnv,
  limit: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.projected_gravity_b[:, 2] > limit


def contact(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  contact_sensor: ContactSensor = env.scene[asset_cfg.name]
  net_contact_forces = contact_sensor.data.net_forces_w_history
  return torch.any(
    torch.max(
      torch.norm(net_contact_forces[:, :, asset_cfg.body_ids], dim=-1),
      dim=1,
    )[0]
    > 1.0,
    dim=1,
  )


def base_orientation(
  env: ManagerBasedRlEnv,
  limit: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.norm(asset.data.projected_gravity_b[:, :2], dim=1) - limit


def n_foot_contact(
  env: ManagerBasedRlEnv,
  number_of_desired_feet: int,
  min_command_value: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  contact_sensor: ContactSensor = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  net_contact_forces = contact_sensor.data.net_forces_w_history
  contact_cstr = torch.abs(
    (
      torch.max(
        torch.norm(net_contact_forces[:, :, asset_cfg.body_ids], dim=-1),
        dim=1,
      )[0]
      > 1.0
    ).sum(1)
    - number_of_desired_feet
  )
  command_more_than_limit = (
    torch.norm(command[:, :3], dim=1) > min_command_value
  ).float()
  return contact_cstr * command_more_than_limit


def joint_range(
  env: ManagerBasedRlEnv,
  limit: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return (
    torch.abs(
      asset.data.joint_pos[:, asset_cfg.joint_ids]
      - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    - limit
  )


def action_rate(
  env: ManagerBasedRlEnv,
  limit: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  return (
    torch.abs(
      env.action_manager._action[:, asset_cfg.joint_ids]
      - env.action_manager._prev_action[:, asset_cfg.joint_ids]
    )
    / env.step_dt
    - limit
  )


def foot_contact_force(
  env: ManagerBasedRlEnv,
  limit: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  contact_sensor: ContactSensor = env.scene[asset_cfg.name]
  net_contact_forces = contact_sensor.data.net_forces_w_history
  return (
    torch.max(torch.norm(net_contact_forces[:, :, asset_cfg.body_ids], dim=-1), dim=1)[
      0
    ]
    - limit
  )


def min_base_height(
  env: ManagerBasedRlEnv,
  limit: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return limit - asset.data.geom_pos_w[:, 2]


def joint_position_limits(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  data: EntityData = env.scene[asset_cfg.name].data
  joint_pos = data.joint_pos[:, asset_cfg.joint_ids]
  lower_violation = data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0] - joint_pos
  upper_violation = joint_pos - data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]

  return torch.maximum(lower_violation, upper_violation)


def joint_velocity_limits(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  data: EntityData = env.scene[asset_cfg.name].data
  joint_vel = data.joint_vel[:, asset_cfg.joint_ids]
  limit = data.joint_vel_limits[:, asset_cfg.joint_ids]
  print(limit, flush=True)
  violation = torch.abs(joint_vel) - limit
  return violation


def joint_torque_limits(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  data: EntityData = env.scene[asset_cfg.name].data
  torques = data.qfrc_actuator[:, asset_cfg.joint_ids]
  limit = data.joint_effort_limits[:, asset_cfg.joint_ids]
  violation = torch.abs(torques) - limit
  return violation


def air_time(
  env: ManagerBasedRlEnv,
  limit: float,
  velocity_deadzone: float,
  command_name: str,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  contact_sensor: ContactSensor = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."

  touchdown = contact_sensor.compute_first_contact(env.step_dt)[:, asset_cfg.body_ids]
  last_air_time = contact_sensor.data.last_air_time[:, asset_cfg.body_ids]

  # Get velocity command and check ALL components against deadzone
  cmd_active = (
    torch.any(
      torch.abs(command) > velocity_deadzone,  # Check x,y,z separately
      dim=1,
    )
    .float()
    .unsqueeze(1)
  )  # Shape: (num_envs, 1)

  # Apply constraint only when command is active (any component > deadzone)
  return (limit - last_air_time) * touchdown.float() * cmd_active


def foot_contact(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  contact_sensor: ContactSensor = env.scene[asset_cfg.name]
  net_contact_forces = contact_sensor.data.net_forces_w_history

  # Compute number of feet in contact per environment
  foot_contacts = (
    torch.max(
      torch.norm(
        net_contact_forces[:, :, asset_cfg.body_ids],
        dim=-1,
      ),
      dim=1,
    )[0]
    > 1.0  # Boolean: (envs, num_feet)
  ).sum(1)  # Sum over feet → (envs,)

  # Penalize cases where number of contacts is not 1 or 2
  contact_cstr = ((foot_contacts < 1) | (foot_contacts > 2)).float()

  return contact_cstr


def no_move(
  env: ManagerBasedRlEnv,
  velocity_deadzone: float,
  joint_vel_limit: float,
  command_name: str,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Constraint that penalizes joint movement when the robot should be stationary.

  Only applies when all components of the base velocity command are within the deadzone.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)[:, :3]
  cmd_inactive_mask = torch.all(torch.abs(command) < velocity_deadzone, dim=1)
  if cmd_inactive_mask.sum() == 0:
    # No env matches — return zero constraint (or any safe fallback)
    return torch.zeros(
      (env.num_envs, sum(env.action_manager.action_term_dim)), device=env.device
    )

  # Filter only relevant environments
  active_joint_vel = asset.data.joint_vel[cmd_inactive_mask][:, asset_cfg.joint_ids]

  # Compute constraint just for those
  cstr_nomove = torch.abs(active_joint_vel) - joint_vel_limit

  # Repeat to match the number of original environments
  num_repeat = env.num_envs // cstr_nomove.shape[0] + 1
  cstr_nomove = cstr_nomove.repeat((num_repeat, 1))[: env.num_envs]

  return cstr_nomove


def foot_orientation(
  env: ManagerBasedRlEnv,
  limit: float,
  desired_projected_gravity: torch.Tensor,
  asset_cfg: SceneEntityCfg,
  sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  foot_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids, :].reshape(-1, 4)
  foot_to_world = torch.transpose(matrix_from_quat(foot_quat_w), dim0=1, dim1=2)
  gravity_vec_foot = torch.matmul(foot_to_world, asset.data.GRAVITY_VEC_W[0].squeeze(0))
  desired_projected_gravity = torch.tensor(
    desired_projected_gravity, dtype=torch.float32, device=env.device
  )
  zero_mask = desired_projected_gravity == 0

  contact_sensor = env.scene[sensor_cfg.name]
  touchdown = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]

  return (
    torch.norm(gravity_vec_foot[:, zero_mask], dim=1).unsqueeze(-1) - limit
  ) * touchdown.float()


def base_height(
  env: ManagerBasedRlEnv,
  height: float,
  std: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  base_height = asset.data.geom_pos_w[:, 2]

  violation = torch.where(
    (base_height < height - std) | (base_height > height + std),
    torch.tensor(1.0, device=base_height.device),
    torch.tensor(0.0, device=base_height.device),
  )

  return violation


def foot_clearance(
  env: ManagerBasedRlEnv,
  min_height: float,
  velocity_deadzone: float,
  command_name: str,
  pos_asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  contact_asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  # Get foot positions from position asset
  pos_asset: Entity = env.scene[pos_asset_cfg.name]
  foot_heights = pos_asset.data.body_link_pos_w[
    :, pos_asset_cfg.body_ids, 2
  ]  # shape: (num_envs, num_feet)
  # Get contact information
  contact_sensor: ContactSensor = env.scene[contact_asset_cfg.name]
  touchdown = contact_sensor.compute_first_contact(env.step_dt)[:, asset_cfg.body_ids]

  # Initialize swing max height tracking if needed
  if not hasattr(pos_asset.data, "swing_max_height"):
    pos_asset.data.swing_max_height = torch.zeros_like(foot_heights)

  # violation
  violation = (min_height - pos_asset.data.swing_max_height.clone()) * touchdown.float()

  # Update max height for feet in swing phase
  pos_asset.data.swing_max_height = torch.where(
    ~touchdown.bool(),
    torch.maximum(pos_asset.data.swing_max_height, foot_heights),
    torch.zeros_like(foot_heights),  # Reset when not in swing
  )

  # Get velocity command and check ALL components against deadzone
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  cmd_active = (
    torch.any(
      torch.abs(command) > velocity_deadzone,  # Check x,y,z separately
      dim=1,
    )
    .float()
    .unsqueeze(1)
  )  # Shape: (num_envs, 1)

  return violation * cmd_active
