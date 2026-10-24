"""teacher_se_v1 fine-tune variant c04 (2026-10-24): teacher_se_v1 with cadence_tracking weight 0.2 -> 0.4.
r27 (relative_scale 0.25 -> 0.10, 50M) left cadence unchanged at 1.78 steps/s (human law 1.66-1.68 at 1 m/s); this
doubles the term instead, for 150M (cadence first followed the command at 150M in r8). Fine-tune from r25.
"""
from examples.experiments.steering.mlp_human_model_v3_teacher_se_v1 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_teacher_se_v1 import env_config as _se_env_config

CADENCE_WEIGHT_C04 = 0.4


def env_config(robot_cfg, args):
    cfg = _se_env_config(robot_cfg, args)
    cfg.reward_components["cadence_tracking"].static_params["weight"] = CADENCE_WEIGHT_C04
    return cfg
