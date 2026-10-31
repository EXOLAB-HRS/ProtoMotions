"""vic4_TC_walkA1ks02_from-walkks02r2_r1 (scratch strategy A, stage A2, candidate 2026-10-31): vic4_TC_walkA1_from-walkr2_r1
from the slot B walker walk_TC_scratch_ks02_from-walkks02r1_r2, so with that walker's single change knee_strike_prior
0.6 -> 0.2 (reason in mlp_human_model_v3_walk_tc_scratch_ks02_r1.py). Everything else is the slot A A2 recipe.
"""
from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walka1_from_walkr2_r1 as _a2
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walka1_from_walkr2_r1 import *  # noqa: F401,F403

KNEE_STRIKE_W = 0.2


def env_config(robot_cfg, args):
    cfg = _a2.env_config(robot_cfg, args)
    cfg.reward_components["knee_strike_prior"].static_params["weight"] = KNEE_STRIKE_W
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _a2.agent_config(robot_cfg, env_cfg, args)
