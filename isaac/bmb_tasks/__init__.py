"""Break My Bot Isaac Lab tasks.

Importing this package only registers gym ids with string entry points, so it is safe before the
Omniverse app is launched (the configs themselves import Isaac Lab and are loaded later by hydra).

- BMB-Go2-Rough-v2          robust policy v2, training (teacher PPO and student distillation)
- BMB-Go2-Rough-v2-Distill  same, friction curriculum already finished (use for stage B)
- BMB-Go2-Rough-v2-Play     small scene, clean observations, no deception / pushes
"""

import gymnasium as gym

_ENV_CFG = f"{__name__}.go2_v2_env_cfg"
_AGENTS = f"{__name__}.agents"


def _register(task_id: str, env_cfg: str) -> None:
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{_ENV_CFG}:{env_cfg}",
            "rsl_rl_cfg_entry_point": f"{_AGENTS}:Go2V2TeacherPPORunnerCfg",
            "rsl_rl_distillation_cfg_entry_point": f"{_AGENTS}:Go2V2DistillationRunnerCfg",
        },
    )


_register("BMB-Go2-Rough-v2", "Go2V2EnvCfg")
_register("BMB-Go2-Rough-v2-Distill", "Go2V2EnvCfg_DISTILL")
_register("BMB-Go2-Rough-v2-Play", "Go2V2EnvCfg_PLAY")
