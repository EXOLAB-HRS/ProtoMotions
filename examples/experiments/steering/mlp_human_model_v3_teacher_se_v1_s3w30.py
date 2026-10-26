"""teacher_se_v1 stage 3, comparison arm (2026-10-25, training setting c): s3 with steady_speed_tracking weight 0.15 -> 0.30.
One change from s3, to see how the steady-state speed correction scales and what it costs the gait.
"""
from examples.experiments.steering.mlp_human_model_v3_teacher_se_v1_s3 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_teacher_se_v1_s3 import env_config as _s3_env_config

STEADY_SPEED_WEIGHT = 0.30


def env_config(robot_cfg, args):
    cfg = _s3_env_config(robot_cfg, args)
    cfg.reward_components["steady_speed_tracking"].static_params["weight"] = STEADY_SPEED_WEIGHT
    return cfg
