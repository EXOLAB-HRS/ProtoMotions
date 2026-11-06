# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 v35 strat spd (2026-11-07): vic4_tc_walk33_v35_strat_r1 with a symmetric steady-speed reward.
v35 strat last met the torque targets (1.0 m/s push-off 1.49 vs WBDS 1.47 N m/kg, late-stance hip -0.36 vs -0.27) and
W2 / W3, but W1 failed on speed: 0.8 m/s +0.09, 1.0 +0.06, 1.4 -0.08 m/s (the speed range is squeezed toward 1.05).
The overspeed tail (weight 0.5, exp(-8 e^2)) gives half credit to overspeed (at +0.09 m/s 0.5 * 0.94), so it forgave
the error it was meant to punish. Here both tails are off: exp(-200 e^2) on either side (sigma 0.05 m/s, the W1 gate),
and the steady weight goes 0.4 -> STEADY_WEIGHT.
"""
from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_v35_strat_r1 as _base
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_v35_strat_r1 import *  # noqa: F401,F403

STEADY_WEIGHT = 0.6


def env_config(robot_cfg, args):
    cfg = _base.env_config(robot_cfg, args)
    sp = cfg.reward_components["steady_speed_tracking"].static_params
    sp.update(weight=STEADY_WEIGHT, overspeed_tail_weight=0.0, underspeed_tail_weight=0.0)
    return cfg
