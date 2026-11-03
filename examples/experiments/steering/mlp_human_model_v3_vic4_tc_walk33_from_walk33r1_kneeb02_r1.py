# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4_TC_walk33_from-walk33r1_kneeb02_r1 (server slot B, 2026-11-03, user approved "B안"): the cleaned law of
vic4_TC_s5_from-s5_ffdistill_r1 (stiffness unfrozen, feed-forward distillation, swing knee out of r_imp, no VIC effort) on
plant human_model_v3.3 from the v3.3 walker, with one change: the sagittal knees keep beta 0.2 of the fixed-PD gap
(applied per physics substep, v4_prior_substep True) while hip / ankle / frontal joints are at beta 0.
Reason: night 11-03 runs at beta 0 collapsed the 1.4 m/s knee (knee r 0.62-0.68, strike 21-23 deg) while beta 0.2 kept it
(0.98); the tau_ff diagnostic put the uncoverable fast torque at the knee only (53-63 % coverable vs >= 90 % hip / ankle).
"""
import dataclasses

from examples.experiments.steering import mlp_human_model_v3_vic4_tc_s5_from_s5_ffdistill_r1 as _base
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_s5_from_s5_ffdistill_r1 import *  # noqa: F401,F403


def env_config(robot_cfg, args):
    cfg = _base.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(
        cfg.control_components["steering"], beta_hold_joints=("L_Knee_y", "R_Knee_y"), beta_hold=0.2, v4_prior_substep=True)
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _base.agent_config(robot_cfg, env_cfg, args)
