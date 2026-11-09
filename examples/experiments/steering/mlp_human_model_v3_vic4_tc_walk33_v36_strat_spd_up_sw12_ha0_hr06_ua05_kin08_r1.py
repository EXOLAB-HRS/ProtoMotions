# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 v36 strat spd up sw12 ha0 hr06 ua05 kin08 (2026-11-10, user: "관절 각도, 토크 궤적이 사람데이터와 맞게", "상체도 좀
모션은 잘맞아야해", "좋은 결과를 내기 위한 설정으로 알아서 다음 학습"): run with --plant v3.6, parent 344 last (V36u300_last).
vic4_tc_walk33_v36_strat_spd_up_sw12_ha0_r1 (run 362 recipe) with the upper-body and GRF / CoP terms raised.
Why (teacher pick 261110, W2-3 / W2-2k at 1.0 m/s, 344 last / 362 e124 / 380 e124):
  head roll ROM 9.7-14.4 deg vs Lencioni 4.6 deg  -> head_roll_stability 0.3 -> 0.6
  arm swing ROM 33-36 deg vs 18.5 deg             -> vic_upper_prior     0.3 -> 0.5
  CoP lever z +1.8..+2.7, braking fx z -1.3..-1.8 -> vic_kinetic_prior   0.5 -> 0.8
Recipe weights only; the reward functions are unchanged.
"""
from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_up_sw12_ha0_r1 as _base
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_up_sw12_ha0_r1 import *  # noqa: F401,F403

HEAD_ROLL_WEIGHT = 0.6
UPPER_WEIGHT = 0.5
KINETIC_WEIGHT = 0.8


def env_config(robot_cfg, args):
    cfg = _base.env_config(robot_cfg, args)
    cfg.reward_components["head_roll_stability"].static_params["weight"] = HEAD_ROLL_WEIGHT
    cfg.reward_components["vic_upper_prior"].static_params["weight"] = UPPER_WEIGHT
    cfg.reward_components["vic_kinetic_prior"].static_params["weight"] = KINETIC_WEIGHT
    return cfg
