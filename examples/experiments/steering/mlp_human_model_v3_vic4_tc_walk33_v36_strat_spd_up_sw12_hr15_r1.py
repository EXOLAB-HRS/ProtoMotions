# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 v36 strat spd up sw12 hr15 (2026-11-08, user: "w1,2,3,2-2k 다 맞추는걸 목표로", "검증생각해서 문제 있는건 개선"): run with --plant v3.6.
vic4_tc_walk33_v36_strat_spd_up_sw12_r1 with head_roll_stability back at 0.15 (one change).
Why: run 352 (344 last + sw12, head roll 0.3) fixed the 0.8 / 1.0 m/s speeds (0.85 / 1.00) and cut early falls (18 -> 1-6 of
64 at 1.4 m/s) and the head roll range (14 -> 9 deg), but the share of strides with positive MoS_ML fell to 0.67-0.75 at
0.8 / 1.0 m/s (344: 0.96-0.98; run 340 with sw12 and no head roll change: >= 0.99) and step width at 1.0 m/s narrowed to
0.14 m. This run tells whether the doubled head roll weight caused the lateral margin loss.
  head_roll_stability weight 0.3 -> 0.15 (= run 344).
"""
from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_up_sw12_r1 as _base
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_up_sw12_r1 import *  # noqa: F401,F403

HEAD_ROLL_WEIGHT = 0.15


def env_config(robot_cfg, args):
    cfg = _base.env_config(robot_cfg, args)
    cfg.reward_components["head_roll_stability"].static_params["weight"] = HEAD_ROLL_WEIGHT
    return cfg
