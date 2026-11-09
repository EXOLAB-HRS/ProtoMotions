# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 v36 strat spd up sw12 wang ha08 tail03 (2026-11-09, server, user: "발목위주로 걷는 사슬을 더 디벨롭 ... 머리 잡는거 ... 보상함수
구성을 전체 다 고려해서 조금 길게 학습을해서 밸런스있게"): run with --plant v3.6, long (80M) from 344 last (ankle push-off chain).
Base vic4_tc_walk33_v36_strat_spd_up_sw12_ha0_wang_r1 (380: 362 setting + WBDS joint-angle table; e124 push-off 1.47 / 1.47,
brake -0.09 / -0.12 BW). Changes, each aimed at a W1 / W2-3 gap of the 344 chain while keeping its torque terms as they are:
  steady_speed_tracking: symmetric wide tails, overspeed_tail_weight 0 -> 0.3 and underspeed_tail_weight 0 -> 0.3, both tail scales 8
    (exp(-8 e^2), sigma ~0.35 m/s). The narrow kernel exp(-200 e^2) gives almost no gradient at the chain's 0.10-0.15 m/s bias
    (e = 0.15: reward 0.011), so more weight (372, steady 2.0) did not move the 0.8 / 1.4 m/s speeds. Symmetric, unlike run 300's
    overspeed-only tail 0.5 that biased toward overspeed.
  head_angular_stability 0 -> 0.08 (head pitch/roll rate; 0.15 lowered MoS in 352/356, ha0 left head pitch free; local 408 uses 0.08),
    head_roll_stability stays 0.3 (a7-headroll term), pelvis_yaw_stability 0.1.
  vic_upper_prior stays 0.3 (arm swing 34-36 deg vs human 18; 0.1 lost W2 in 376).
  Kept: vic_moment_prior 1.0, vic_kinetic_prior 0.5, vic_strategy_cost 0.3 (0.9 collapsed speed in 350 / server 262), steady 1.2.
"""
from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_up_sw12_ha0_wang_r1 as _base
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_up_sw12_ha0_wang_r1 import *  # noqa: F401,F403

HEAD_RATE_WEIGHT = 0.08
TAIL_WEIGHT = 0.3
TAIL_SCALE = 8.0


def env_config(robot_cfg, args):
    cfg = _base.env_config(robot_cfg, args)
    cfg.reward_components["head_angular_stability"].static_params["weight"] = HEAD_RATE_WEIGHT
    p = cfg.reward_components["steady_speed_tracking"].static_params
    p["overspeed_tail_weight"] = TAIL_WEIGHT
    p["underspeed_tail_weight"] = TAIL_WEIGHT
    p["overspeed_tail_scale"] = TAIL_SCALE
    p["underspeed_tail_scale"] = TAIL_SCALE
    return cfg
