"""vic_TC_s3_r16 (implementation vic_tc_v1 on vic_tc_v1_se, candidate, 2026-10-27): vic_TC_s3_r13 continued at alpha 1,
beta 0 (human stiffness 100 %), actor lr 2e-5 (launcher), with the commanded natural cadence set to the human law used by
the W2 judge (check_criteria.py, WBDS / Fukuchi 2018): f0 = 0.558 v + 1.060 steps/s instead of the pack law
0.608 v + 1.178. r12b / r13 / r14 walked at the pack law (1.78 steps/s at 1.0 m/s vs human 1.62), which alone fails W2
cadence_err (0.10 - 0.15). The cadence ratio range (0.8 - 1.25) and every other setting are r14's.
"""
import dataclasses

from examples.experiments.steering.mlp_human_model_v3_vic_tc_s3_r14 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_vic_tc_s3_r14 import (
    env_config as _r14_env_config, agent_config)  # noqa: F401

HUMAN_CADENCE_SLOPE = 0.558
HUMAN_CADENCE_INTERCEPT = 1.060


def env_config(robot_cfg, args):
    cfg = _r14_env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(
        cfg.control_components["steering"], cadence_f0_slope=HUMAN_CADENCE_SLOPE,
        cadence_f0_intercept=HUMAN_CADENCE_INTERCEPT)
    return cfg
