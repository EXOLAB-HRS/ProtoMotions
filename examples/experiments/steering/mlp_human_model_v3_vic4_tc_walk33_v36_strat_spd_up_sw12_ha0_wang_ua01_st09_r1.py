# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 v36 strat spd up sw12 ha0 wang ua01 st09 (2026-11-09, server, user: "w1,2,3,w2-2k 다 골고루 맞춘거 ... 새로 학습 하나 올려주라"): run with --plant v3.6.
vic4_tc_walk33_v36_strat_spd_up_sw12_ha0_wang_ua01_r1 (local 392) with the strategy cost raised.
Why: 392 last passes W1 at 0.8 / 1.0 m/s, W2 v5 2/3 and W3, but its torque is still hip-dominant (1.0 m/s push-off 1.21 / 1.47,
hip -0.49 / -0.27 N.m/kg). Strategy cost 0.9 failed before only together with a speed-suppressing term: 350 (steady 0.6,
arm prior 0.3) walked slower and slower, 354 (arm prior 0.3) stayed compressed. 392 removed the arm-prior suppression
and keeps steady 1.2, so this tests whether the torque gap closes without losing speed.
  vic_strategy_cost weight 0.3 -> 0.9.
"""
from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_up_sw12_ha0_wang_ua01_r1 as _base
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_up_sw12_ha0_wang_ua01_r1 import *  # noqa: F401,F403

STRATEGY_WEIGHT = 0.9


def env_config(robot_cfg, args):
    cfg = _base.env_config(robot_cfg, args)
    cfg.reward_components["vic_strategy_cost"].static_params["weight"] = STRATEGY_WEIGHT
    return cfg
