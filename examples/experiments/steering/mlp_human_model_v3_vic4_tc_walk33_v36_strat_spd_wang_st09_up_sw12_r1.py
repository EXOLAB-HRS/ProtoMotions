# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 v36 strat spd wang st09 up sw12 (2026-11-08, user: "w1,2,3,2-2k 다 맞추는걸 목표로", "새로운 학습설정 만들어서 계속
돌려놔줘"): run with --plant v3.6. vic4_tc_walk33_v36_strat_spd_wang_st09_up_r1 (strategy cost 0.9 + arm / head terms) with
the speed and head-roll weights of vic4_tc_walk33_v36_strat_spd_up_sw12_r1.
Why: run 340 e483 (wang + steady 1.2) passes W2 v5 and W3 and misses W1 only at 1.4 m/s (-0.058 m/s), but walks with the hip
(1.0 m/s push-off 1.13 vs 1.47, late-stance hip -0.52 vs -0.27 N m/kg) and has no arm / head terms. Continuing it with the
strategy cost 0.9 and the arm / head terms tests whether the torque and the upper body move to human while W1 holds.
  steady_speed_tracking 0.6 -> 1.2; speed_transition_tracking overspeed tail 0.5 -> 0.0; head_roll_stability 0.15 -> 0.3.
"""
from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_wang_st09_up_r1 as _base
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_wang_st09_up_r1 import *  # noqa: F401,F403

STEADY_WEIGHT = 1.2
HEAD_ROLL_WEIGHT = 0.3


def env_config(robot_cfg, args):
    cfg = _base.env_config(robot_cfg, args)
    cfg.reward_components["steady_speed_tracking"].static_params["weight"] = STEADY_WEIGHT
    cfg.reward_components["speed_transition_tracking"].static_params["overspeed_tail_weight"] = 0.0
    cfg.reward_components["head_roll_stability"].static_params["weight"] = HEAD_ROLL_WEIGHT
    return cfg
