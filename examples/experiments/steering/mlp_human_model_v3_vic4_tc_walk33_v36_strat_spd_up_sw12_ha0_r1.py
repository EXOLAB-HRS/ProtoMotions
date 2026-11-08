# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 v36 strat spd up sw12 ha0 (2026-11-08, user: "w1,2,3,2-2k를 만족할만한 후보", "검증생각해서 문제 있는건 개선"): run with
--plant v3.6. vic4_tc_walk33_v36_strat_spd_up_sw12_r1 with the head angular-rate term off.
Why: the MoS_ML positive share drops only on the arm/head ("up") chain under steady 1.2: run 352 last 0.67-0.75 and run 356
(head roll 0.15) last 0.77-0.92, while run 340 (steady 1.2, no up terms) stays >= 0.99 at the same 0.8 m/s walked speed.
Head roll 0.3 vs 0.15 did not decide it (356), so this run removes the other head term, head_angular_stability 0.15 -> 0
(it penalises the head pitch / roll rate and so the trunk sway used to move the CoM over the stance foot); vic_upper_prior
(arms) stays. If MoS recovers the head-rate term is the cause; if not, the arm prior is next.
"""
from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_up_sw12_r1 as _base
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_up_sw12_r1 import *  # noqa: F401,F403

HEAD_RATE_WEIGHT = 0.0


def env_config(robot_cfg, args):
    cfg = _base.env_config(robot_cfg, args)
    cfg.reward_components["head_angular_stability"].static_params["weight"] = HEAD_RATE_WEIGHT
    return cfg
