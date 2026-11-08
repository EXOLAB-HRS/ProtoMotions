# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 v36 strat spd up sw12 ha0 ua01 (2026-11-08, user: "w1,2,3,2-2k 다 맞추는걸 목표로"): run with --plant v3.6.
vic4_tc_walk33_v36_strat_spd_up_sw12_ha0_r1 with less weight on the arm prior.
Why: ha0 passes W2 v5 / W3 with torque near human on two seeds (362, 366) but the speed range stays compressed (1.4 m/s
command walked at 1.22-1.29). More steady-speed weight did not help (372, steady 2.0: 1.4 m/s still 1.19-1.21, W2 failed).
Runs without the upper-body terms reach the speed range (340 / 358, bias <= 0.044); with them it compresses (344, 354).
  vic_upper_prior weight 0.3 -> 0.1.
"""
from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_up_sw12_ha0_r1 as _base
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_up_sw12_ha0_r1 import *  # noqa: F401,F403

UPPER_WEIGHT = 0.1


def env_config(robot_cfg, args):
    cfg = _base.env_config(robot_cfg, args)
    cfg.reward_components["vic_upper_prior"].static_params["weight"] = UPPER_WEIGHT
    return cfg
