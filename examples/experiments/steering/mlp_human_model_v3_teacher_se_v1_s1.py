"""teacher_se_v1 stage 1 (2026-10-24): same recipe as teacher_se_v1 with a narrow command set for the first part
of the from-scratch run, as for a7 (the root of every teacher; broad-command scratch runs a3-a5 and P2/P3 never
learned stops or natural gait): 0.8-1.2 m/s, no stops, no turns, natural cadence only (rho* = 1).
Stage 2 continues from this run's last.ckpt with mlp_human_model_v3_teacher_se_v1 (r25's full command set).
"""
from examples.experiments.steering.mlp_human_model_v3_teacher_se_v1 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_teacher_se_v1 import env_config as _se_env_config


def env_config(robot_cfg, args):
    cfg = _se_env_config(robot_cfg, args)
    s = cfg.control_components["steering"]
    s.tar_speed_min, s.tar_speed_max = 0.8, 1.2
    s.stop_probability = 0.0
    s.turn_fraction = 0.0
    s.speed_anchor_targets, s.speed_anchor_probability = (0.8, 1.0, 1.2), 0.5
    s.cadence_natural_fraction = 1.0
    return cfg
