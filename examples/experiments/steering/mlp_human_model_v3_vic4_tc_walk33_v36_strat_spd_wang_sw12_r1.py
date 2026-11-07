# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 v36 strat spd wang sw12 (2026-11-07, user: "w1, w2, w3는 기존처럼 잘 통과하면서", "이어서 새로 학습 설정해서 계속 돌려줘"):
run with --plant v3.6. Same as vic4_tc_walk33_v36_strat_spd_wang_r1, with more weight on speed tracking only.
Why: run 300 (v35 strat spd, steady tail off) still compressed the speed range like v35 strat (last 0.8 / 1.4 m/s commands
walked at 0.90 / 1.23 m/s; e276 0.78 / 1.31), so the overspeed tail was not the cause; the moment / kinetic / strategy
priors (indexed by the commanded speed, total weight 1.8) outweigh the speed terms (steady 0.6 + transition 0.3) as training goes on.
  steady_speed_tracking weight 0.6 -> 1.2; speed_transition_tracking overspeed tail 0.5 -> 0.0 (no credit for overspeed).
"""
from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_wang_r1 as _base
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_wang_r1 import *  # noqa: F401,F403

STEADY_WEIGHT = 1.2


def env_config(robot_cfg, args):
    cfg = _base.env_config(robot_cfg, args)
    cfg.reward_components["steady_speed_tracking"].static_params["weight"] = STEADY_WEIGHT
    cfg.reward_components["speed_transition_tracking"].static_params["overspeed_tail_weight"] = 0.0
    return cfg
