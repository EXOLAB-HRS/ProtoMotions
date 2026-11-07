# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 v36 strat spd wang st09 up (2026-11-07, user: "머리, 팔쪽도 고려해서 ㄱㄱ"): vic4_tc_walk33_v36_strat_spd_wang_st09_r1
plus the upper-body terms of vic4_tc_walk33_v36_strat_spd_up_r1 (vic_upper_prior 0.3, head_angular_stability 0.15)."""
from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_wang_st09_r1 as _base
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_wang_st09_r1 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_up_r1 import add_upper


def env_config(robot_cfg, args):
    return add_upper(_base.env_config(robot_cfg, args), robot_cfg)
