"""vic_TC_r5 (implementation vic_tc_v1, candidate change, 2026-10-24): vic_TC_r3 with the metabolic proxy weight halved.
Starts from r1 last.ckpt like r2-r4.

r2 W1-W4 check (output/261023/e02_vic_tc_r2_loop/evaluation/criteria_vic_tc_r2.json): the metabolic proxy at 0.03 put
the energy ratio inside the Collins limit (10 N m 3.4-3.6, r1 4.0-4.5) but walking got worse than r1: falls 0.43/min,
S_walk15 0.82, ankle curve r 0.32. r5 keeps r3's gait terms (human_gait_prior x1.6, cadence_tracking 0.10) and sets
vic_metab to METAB_WEIGHT, between r1 (none) and r2 (0.03).
"""
from examples.experiments.steering.mlp_human_model_v3_vic_tc_r3 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_vic_tc_r3 import (
    env_config as _r3_env_config, agent_config)  # noqa: F401

METAB_WEIGHT = 0.015


def env_config(robot_cfg, args):
    cfg = _r3_env_config(robot_cfg, args)
    cfg.reward_components["vic_metab"].static_params["weight"] = METAB_WEIGHT
    return cfg
