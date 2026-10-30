"""walk_TC_scratch_ks02_from-walkks02r1_r2 (2026-10-30, slot B; A1b of the knee_strike_prior 0.2 chain): from
walk_TC_scratch_ks02_r1 last, walk_TC_scratch_from-walkr1_r2 (= teacher_se_v1 s5 recipe, 350M) with the same single
change knee_strike_prior 0.6 -> 0.2 (reason in mlp_human_model_v3_walk_tc_scratch_ks02_r1.py).
"""
from examples.experiments.steering.mlp_human_model_v3_walk_tc_scratch_from_walkr1_r2 import *  # noqa: F401,F403
from examples.experiments.steering import mlp_human_model_v3_walk_tc_scratch_from_walkr1_r2 as _a1b

KNEE_STRIKE_W = 0.2


def env_config(robot_cfg, args):
    cfg = _a1b.env_config(robot_cfg, args)
    cfg.reward_components["knee_strike_prior"].static_params["weight"] = KNEE_STRIKE_W   # as teacher_cadence_v1 r22
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _a1b.agent_config(robot_cfg, env_cfg, args)
