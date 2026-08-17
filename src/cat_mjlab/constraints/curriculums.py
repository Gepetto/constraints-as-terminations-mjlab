"""Common functions that can be used to create curriculum for the learning environment."""

from __future__ import annotations

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from cat_mjlab.envs.cat_env import CaTManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


def modify_constraint_p(
  env: CaTManagerBasedRlEnv,
  env_ids: torch.Tensor,
  term_name: str,
  num_steps: int,
  init_max_p: float,
):
  assert isinstance(env, CaTManagerBasedRlEnv), (
    "Constraint term 'modify_constraint_p' requires 'env' to be of a type 'CaTManagerBasedRlEnv'"
  )
  progress = min(env.common_step_counter / num_steps, 1.0)

  # Linearly interpolate the expected time for episode end: soft_p is the maximum
  # termination probability so it is an image of the expected time of death.
  T_start = 20
  T_end = 1.0 / init_max_p
  init_max_p = 1.0 / (T_start + progress * (T_end - T_start))

  # obtain term settings
  term_cfg = env.constraint_manager.get_term_cfg(term_name)
  # update term settings
  term_cfg.max_p = init_max_p
  env.constraint_manager.set_term_cfg(term_name, term_cfg)

  return init_max_p
