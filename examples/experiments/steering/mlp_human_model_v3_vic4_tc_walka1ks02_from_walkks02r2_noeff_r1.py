"""vic4_TC_walkA1ks02_from-walkks02r2_noeff_r1 (scratch strategy A stage A2 candidate, 2026-11-01): vic4_TC_walkA1ks02_from-
walkks02r2_r1 with the VIC effort cost removed (reason in mlp_human_model_v3_vic4_tc_s5_from_s5_noeff_r1.py). Same parent
(slot B walker walk_TC_scratch_ks02_from-walkks02r1_r2 last); the pair with the s5 run separates parent and effort cost.
"""
from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walka1ks02_from_walkks02r2_r1 as _a2
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walka1ks02_from_walkks02r2_r1 import *  # noqa: F401,F403


def env_config(robot_cfg, args):
    cfg = _a2.env_config(robot_cfg, args)
    cfg.reward_components.pop("vic_effort")
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _a2.agent_config(robot_cfg, env_cfg, args)
