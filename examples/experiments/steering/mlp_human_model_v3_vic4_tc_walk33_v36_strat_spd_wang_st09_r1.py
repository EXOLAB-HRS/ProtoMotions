# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 v36 strat spd wang st09 (2026-11-07, user: "푸시오프랑 엉덩이 토크는 범위에 들어 왔나? 다음학습도 준비하자"):
vic4_tc_walk33_v36_strat_spd_wang_r1 with the strategy cost weight 0.3 -> 0.9 (recipe weight only). Run 325 (v3.6) fixed
the speed range (1.4 m/s command -> 1.36-1.39 m/s, W1 v3 on 2 of 3 rollout seeds at e276) but walks with the hip:
at 1.0 m/s push-off 1.14 / 1.24 / 1.40 vs human 1.47 / 1.47 / 1.51 N m/kg and late-stance hip minimum -0.83 / -0.57 / -0.51
vs -0.27 / -0.27 / -0.30 N m/kg (e276 / e483 / last), braking fx -0.31 vs -0.12 BW at e276.
"""
from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_wang_r1 as _base
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_wang_r1 import *  # noqa: F401,F403

STRATEGY_WEIGHT = 0.9


def env_config(robot_cfg, args):
    cfg = _base.env_config(robot_cfg, args)
    cfg.reward_components["vic_strategy_cost"].static_params["weight"] = STRATEGY_WEIGHT
    return cfg
