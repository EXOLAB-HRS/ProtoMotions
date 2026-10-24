"""vic_TC_r10 (implementation vic_tc_v1, candidate change, 2026-10-24): vic_TC_r9 with the baseline-gait anchor at
weight 0.50 instead of 0.25 (r9 vs r10 = anchor strength). Starts from r1 last.ckpt.
"""
from examples.experiments.steering.mlp_human_model_v3_vic_tc_r9 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_vic_tc_r9 import anchored_env_config

ANCHOR_WEIGHT = 0.50


def env_config(robot_cfg, args):
    return anchored_env_config(robot_cfg, args, ANCHOR_WEIGHT)
