# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 v36 strat spd up sw20 ha0 wang (2026-11-09, user: "w1,2,3,2-2k 다 맞추는걸 목표로"): run with --plant v3.6.
vic4_tc_walk33_v36_strat_spd_up_sw12_ha0_wang_r1 with more weight on steady speed tracking.
Why: from the spdw1 e120 parent (386, s2947) this recipe passes W2 v5 and W3 on all seeds with MoS_ML positive share 1.00
at e322, and fails W1 only on speed: every command is walked too slowly (0.8 / 1.0 / 1.4 -> 0.55 / 0.78 / 1.24 m/s).
From the compressed 344 parent steady 2.0 did not help (372), but there the range was compressed both ways; here the
error is a uniform undershoot.
  steady_speed_tracking weight 1.2 -> 2.0.
"""
from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_up_sw12_ha0_wang_r1 as _base
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_up_sw12_ha0_wang_r1 import *  # noqa: F401,F403

STEADY_WEIGHT = 2.0


def env_config(robot_cfg, args):
    cfg = _base.env_config(robot_cfg, args)
    cfg.reward_components["steady_speed_tracking"].static_params["weight"] = STEADY_WEIGHT
    return cfg
