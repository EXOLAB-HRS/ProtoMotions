# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 v36 strat spd up sw12 (2026-11-08, user: "w1,2,3,2-2k 다 맞추는걸 목표로", "머리, 팔쪽도 고려해서 ㄱㄱ"): run with --plant v3.6.
vic4_tc_walk33_v36_strat_spd_up_r1 with more weight on speed tracking and head roll.
Why: run 344 (up, from 300 e483) passed W2 v5 (0.81-0.90) and W3 on all epochs and brought the arm swing range at 1.0 m/s
from 56 to 34 deg (human 18 +- 7), but W1 failed on speed (last: 0.8 / 1.0 / 1.4 m/s walked at 0.95 / 1.11 / 1.29, the
speed range compressed as in run 300) and the head roll range grew to 14 deg (human 4.6 +- 2.2). Run 340 showed that
steady 1.2 + no overspeed credit restores the speed range (W1 passed at e276 s23).
  steady_speed_tracking weight 0.6 -> 1.2; speed_transition_tracking overspeed tail 0.5 -> 0.0 (as run 340);
  head_roll_stability weight 0.15 -> 0.3.
"""
from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_up_r1 as _base
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_up_r1 import *  # noqa: F401,F403

STEADY_WEIGHT = 1.2
HEAD_ROLL_WEIGHT = 0.3


def env_config(robot_cfg, args):
    cfg = _base.env_config(robot_cfg, args)
    cfg.reward_components["steady_speed_tracking"].static_params["weight"] = STEADY_WEIGHT
    cfg.reward_components["speed_transition_tracking"].static_params["overspeed_tail_weight"] = 0.0
    cfg.reward_components["head_roll_stability"].static_params["weight"] = HEAD_ROLL_WEIGHT
    return cfg
