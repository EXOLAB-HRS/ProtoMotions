"""teacher_cadence_v1 revision 6: wide cadence-tracking kernel, smoother measurement.

The r4 sweep trace (2026-10-04 reward check) showed the r3-r5 cadence rewards are flat far
from the natural cadence: with relative scale 0.04, cadence_tracking is 0.000 at rho* = 0.8
and 1.25 and a 5% move toward the target gains nothing; the clock reward averages ~0 once
the phase drifts. Changes from r4 (r3 recipe + random actor cadence columns, actor lr 2e-5):
- cadence_tracking relative scale 0.04 -> 0.25 (a 5% move gains +0.20), weight 0.12 -> 0.20;
- cadence_phase_contact_signed weight 0.18 -> 0.10 (cadence share stays 0.30, heading 0.15);
- measured cadence averages 4 step intervals instead of 2 (control-step quantisation ~3%).
"""
from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r3 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r3 import (
    env_config as _r3_env_config, agent_config)  # noqa: F401

TRACKING_WEIGHT_R6, TRACKING_SCALE_R6 = 0.20, 0.25
PHASE_WEIGHT_R6 = 0.10
CADENCE_INTERVALS = 4


def env_config(robot_cfg, args):
    cfg = _r3_env_config(robot_cfg, args)
    cfg.control_components["steering"].cadence_intervals = CADENCE_INTERVALS
    tracking = cfg.reward_components["cadence_tracking"].static_params
    tracking["weight"], tracking["relative_scale"] = TRACKING_WEIGHT_R6, TRACKING_SCALE_R6
    cfg.reward_components["cadence_phase_contact"].static_params["weight"] = PHASE_WEIGHT_R6
    return cfg
