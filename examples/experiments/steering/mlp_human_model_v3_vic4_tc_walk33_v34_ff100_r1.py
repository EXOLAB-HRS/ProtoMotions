# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 v34 ff100 (2026-11-05, server slots A/B, user: "발목 pushoff 많이 쓰는 방향으로 ㄱㄱ"): the v3.4-only W1 + W2 + W3
recipe (vic4_tc_walk33_band1spd_lp15_swk_spdw1_r1 on --plant v3.4, parent cadence) with ONE change: tau_res_scale
0.33 -> 1.0, so the feed-forward (active muscle) torque may reach the joint strength cap instead of a third of it.
Torque diagnosis of v3.4-only e161 at 1.0 m/s: ankle push-off 0.76 / 0.63 N m/kg at 40 / 50 % vs human 1.16 / 1.47, ankle
tau_ff ~ 0 because its cap was 0.33 x 118 N m = 0.62 N m/kg; the hip compensates with a large late-stance flexion moment
(-0.58 to -0.99 vs +0.07 to -0.24) and the knee stays straight with the opposite moment sign in mid stance.
"""
import dataclasses

from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_band1spd_lp15_swk_spdw1_r1 as _base
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_band1spd_lp15_swk_spdw1_r1 import *  # noqa: F401,F403


def env_config(robot_cfg, args):
    cfg = _base.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(cfg.control_components["steering"], tau_res_scale=1.0)
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _base.agent_config(robot_cfg, env_cfg, args)
