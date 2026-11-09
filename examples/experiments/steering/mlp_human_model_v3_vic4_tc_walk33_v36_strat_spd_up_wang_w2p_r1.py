# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 v36 strat spd up wang w2p (2026-11-09, server, user: "w1이 완벽하게 맞지는 않더라도 w2쪽 관절궤적, 관절토크가 사람데이터에
더 잘맞는게 중요 ... 논문화 할때 검증데이터 ... weight을 이렇게 좀 두고 진행하도록 하자 앞으로"): run with --plant v3.6, long (100M)
from 344 last (ankle push-off chain). W2 priority: joint trajectories and joint torques over exact speed tracking.
Base vic4_tc_walk33_v36_strat_spd_up_sw12_wang_ha08_tail03_r1 (WBDS joint-angle table, symmetric wide steady-speed tails 0.3 / scale 8,
head roll 0.3, head rate 0.08, arm prior 0.3), then:
  human_gait_prior (joint angles, WBDS train5)  0.25 -> 0.6
  vic_moment_prior (joint moments, WBDS train5)  1.0 -> 1.5
  steady_speed_tracking                          1.2 -> 0.8 (tails kept; not below 0.6: steady 0.6 + strategy 0.9 slowed to a stop in 350)
  Kept: vic_kinetic_prior 0.5, vic_strategy_cost 0.3, AMP 0.4 / task 0.6, upper-body terms.
"""
from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_up_sw12_wang_ha08_tail03_r1 as _base
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_up_sw12_wang_ha08_tail03_r1 import *  # noqa: F401,F403

ANGLE_WEIGHT = 0.6
MOMENT_WEIGHT = 1.5
STEADY_WEIGHT = 0.8


def env_config(robot_cfg, args):
    cfg = _base.env_config(robot_cfg, args)
    cfg.reward_components["human_gait_prior"].static_params["weight"] = ANGLE_WEIGHT
    cfg.reward_components["vic_moment_prior"].static_params["weight"] = MOMENT_WEIGHT
    cfg.reward_components["steady_speed_tracking"].static_params["weight"] = STEADY_WEIGHT
    return cfg
