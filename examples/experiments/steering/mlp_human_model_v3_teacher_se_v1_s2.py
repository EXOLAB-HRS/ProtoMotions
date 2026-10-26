"""teacher_se_v1 stage 2 (2026-10-24): teacher_se_v1 (r25's full command set) with reset_speed_from_spawn, for the
second part of the from-scratch run (continues from stage 1 last.ckpt). On reset the ramping command starts at the
spawn frame's root speed instead of 1 m/s (r25: ~10 % of starts fell within 3 s of a reset, mostly a stop goal after a
moving AMASS spawn). Stage 1 had no stops, so the flag only matters here.
"""
from examples.experiments.steering.mlp_human_model_v3_teacher_se_v1 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_teacher_se_v1 import env_config as _se_env_config


def env_config(robot_cfg, args):
    cfg = _se_env_config(robot_cfg, args)
    cfg.control_components["steering"].reset_speed_from_spawn = True
    return cfg
