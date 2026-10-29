"""vic_TC_s5_r17 (implementation vic_tc_v1 on vic_tc_v1_se, candidate, 2026-10-27): vic_TC_s5_r15 continued 100M at
alpha 1, beta 0 (human stiffness 100 %), actor lr 2e-5, with the commanded natural cadence set to the human law of the W2
judge (check_criteria.py, WBDS / Fukuchi 2018): f0 = 0.558 v + 1.060 steps/s instead of the pack law 0.608 v + 1.178.
r15 passes W3 and W4 and fails only lateral 0.079, upright 0.938, cadence 0.156 (walks the pack law, 1.78 steps/s at
1.0 m/s vs human 1.62) and ankle r 0.795. alpha / beta are read from r15's state file (1.0 / 0.0); everything else is r15.
"""
import dataclasses
import json
import os

from examples.experiments.steering.mlp_human_model_v3_vic_tc_s5_r15 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_vic_tc_s5_r15 import (
    env_config as _r15_env_config, agent_config)  # noqa: F401

R15_STATE = os.environ.get(
    "HC_VIC_RESUME_STATE",
    "/home/ghlee/human-controller/output/261023/e02_vic_tc_r2_loop/vic-tc-s5-r15.vic_state.json")
HUMAN_CADENCE_SLOPE = 0.558
HUMAN_CADENCE_INTERCEPT = 1.060


def env_config(robot_cfg, args):
    cfg = _r15_env_config(robot_cfg, args)
    s = json.load(open(R15_STATE))
    cfg.control_components["steering"] = dataclasses.replace(
        cfg.control_components["steering"], alpha_init=float(s["alpha"]), beta_init=float(s["beta"]), schedule=True,
        cadence_f0_slope=HUMAN_CADENCE_SLOPE, cadence_f0_intercept=HUMAN_CADENCE_INTERCEPT)
    return cfg
