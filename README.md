# CaT Mjlab

[![mjlab](https://img.shields.io/badge/mjlab-1.1.1-76B900.svg)](https://mujocolab.github.io/mjlab/v1.1.1/index.html)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-BSD%202--Clause-blue.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Reimplementation of Constraints as Termination algorithm from [Isaac Lab](https://github.com/Gepetto/constraints-as-terminations) to Mjlab.

## Create Environment

```python
import cat_mjlab.constraints as mdp_constraints
from cat_mjlab.envs import CaTManagerBasedRlEnvCfg
from cat_mjlab.managers.constraint_manager import ConstraintTermCfg

def my_cat_env_cfg()

  ...

  constraints = {
    "joint_velocity_limits": ConstraintTermCfg(
       func=mdp_constraints.joint_velocity_limits,
       max_p=1.0,
       params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    ),
  }

  curriculum = {
    "joint_velocity_limits": CurriculumTermCfg(
      func=mdp_constraints.modify_constraint_p,
      params={
        "term_name": "joint_velocity_limits",
        "num_steps": MY_NUM_STEPS,
        "init_max_p": 0.25,
      },
    ),
  }

  return CaTManagerBasedRlEnvCfg(
    ...
    constraints=constraints,
    curriculum=curriculum,
  )
```

## Configure RL

```python
def my_ppo_cat_runner_cfg():
  return RslRlOnPolicyRunnerCfg(
    ...
    algorithm=RslRlPpoAlgorithmCfg(
      ...
      class_name="cat_mjlab.rsl_rl.cat_ppo.CaTPPO"
    )
  )
```

## Register Task

```python
register_mjlab_task(
  task_id="My-CaT-Task",
  env_cfg=my_cat_env_cfg(),
  play_env_cfg=my_cat_env_cfg(play=True),
  rl_cfg=my_ppo_cat_runner_cfg(),
  runner_cls=MjlabOnPolicyRunner,
)
```
```
```

## Train

> [!IMPORTANT]
> Currently (as of version 1.4.0) mjlab does not support custom environments. That is why we have to use script `train-cat` which enables this functionality. This script is also compatible with standard `ManagerBasedRlEnv`


```bash
uv run train-cat My-CaT-Task
```



## Citing

Please cite this work as:

```
@inproceedings{chane2024cat,
      title={CaT: Constraints as Terminations for Legged Locomotion Reinforcement Learning},
      author={Elliot Chane-Sane and Pierre-Alexandre Leziart and Thomas Flayols and Olivier Stasse and Philippe Sou{\`e}res and Nicolas Mansard},
      booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
      year={2024}
}
```
