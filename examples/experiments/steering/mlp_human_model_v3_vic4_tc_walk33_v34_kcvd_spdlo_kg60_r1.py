# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 v34 kcvd spdlo kg60 (2026-11-05 local night): kcvd spdlo with the knee speed exponent 0.87 -> 0.60.
Why: with gamma 0.87 the centre ratio 1.4 / 0.8 m/s is 1.75^0.87 = 1.63 and the learned stance knee k_eff ratio was 1.70-1.72,
above the human 1.37-1.54 (WBDS); hip (1.43) and ankle (1.35) were inside. 1.75^0.60 = 1.40, plus the ~4 % the policy added,
gives about 1.45, the middle of the human range. Hip and ankle exponents unchanged.
"""
import dataclasses

from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_v34_kcvd_spdlo_r1 as _base
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_v34_kcvd_spdlo_r1 import *  # noqa: F401,F403

KC_SPEED_GAMMA = {"Hip_y": 0.64, "Knee_y": 0.60, "Ankle_y": 0.52}


def env_config(robot_cfg, args):
    cfg = _base.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(cfg.control_components["steering"],
                                                             v4_kc_speed_gamma=dict(KC_SPEED_GAMMA))
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _base.agent_config(robot_cfg, env_cfg, args)
