# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 v36 strat spd up sw20 ha0 (2026-11-08, user: "w1,2,3,2-2k 다 맞추는걸 목표로"): run with --plant v3.6.
vic4_tc_walk33_v36_strat_spd_up_sw12_ha0_r1 with more weight on steady speed tracking.
Why: ha0 passed W2 v5 and W3 at e124 on both seeds (362 s2947, 366 s2948) with torque near human, but the speed range stays
compressed (366 e124: 0.8 / 1.0 / 1.4 m/s walked at 0.91 / 1.04 / 1.22; 362 e124 0.95 / 1.12 / 1.29) from the compressed
parent (344 last). Run 340 / 358 (steady 1.2, no upper-body terms) reached bias <= 0.044, so the upper-body terms pull
against speed tracking and steady 1.2 is not enough here.
  steady_speed_tracking weight 1.2 -> 2.0.
"""
from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_up_sw12_ha0_r1 as _base
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_up_sw12_ha0_r1 import *  # noqa: F401,F403

STEADY_WEIGHT = 2.0


def env_config(robot_cfg, args):
    cfg = _base.env_config(robot_cfg, args)
    cfg.reward_components["steady_speed_tracking"].static_params["weight"] = STEADY_WEIGHT
    return cfg
