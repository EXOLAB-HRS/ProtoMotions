# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 v34 kcvd (2026-11-05 night batch, user: "미해결 문제 최대한 해결할수 있는 방향으로"): the v3.4-only W1 + W2 + W3 run
(recipe vic4_tc_walk33_band1spd_lp15_swk_spdw1_r1 on --plant v3.4, cadence as the parent) plus the speed-dependent stance
stiffness and damping centres of vic4_tc_walk33_v34_cadh_kcvd_r1, WITHOUT the human cadence law. The cadh runs collapsed the
knee at one speed per checkpoint while the v3.4-only run kept knee r >= 0.98, so the speed centres are tested here alone.
"""
import dataclasses

from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_band1spd_lp15_swk_spdw1_r1 as _base
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_band1spd_lp15_swk_spdw1_r1 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_v34_cadh_kcv_r1 import KC_SPEED_GAMMA


def env_config(robot_cfg, args):
    cfg = _base.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(cfg.control_components["steering"],
                                                             v4_kc_speed_gamma=dict(KC_SPEED_GAMMA), v4_kc_speed_damp=True)
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _base.agent_config(robot_cfg, env_cfg, args)
