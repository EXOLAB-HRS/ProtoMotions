"""vic4c_TC_scratch_r1 (scratch strategy A, stage A1a, candidate 2026-10-30): from scratch, the A1b recipe
(vic4c_TC_scratch_from-vic4cr1_r2: v4 law frozen at the human impedance centre, s5 rewards, no afferent delay) with the
narrow first-stage commands of teacher_se_v1_s1 (as a7, the root of every teacher; broad-command scratch runs a3-a5 and
P2 / P3 never learned stops or natural gait): 0.8-1.2 m/s, no stops, no turns, natural cadence only (rho* = 1).
"""
from examples.experiments.steering import mlp_human_model_v3_vic4c_tc_scratch_from_vic4cr1_r2 as _a1b
from examples.experiments.steering.mlp_human_model_v3_vic4c_tc_scratch_from_vic4cr1_r2 import *  # noqa: F401,F403


def env_config(robot_cfg, args):
    cfg = _a1b.env_config(robot_cfg, args)
    s = cfg.control_components["steering"]
    s.tar_speed_min, s.tar_speed_max = 0.8, 1.2
    s.stop_probability = 0.0
    s.turn_fraction = 0.0
    s.speed_anchor_targets, s.speed_anchor_probability = (0.8, 1.0, 1.2), 0.5
    s.cadence_natural_fraction = 1.0
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _a1b.agent_config(robot_cfg, env_cfg, args)
