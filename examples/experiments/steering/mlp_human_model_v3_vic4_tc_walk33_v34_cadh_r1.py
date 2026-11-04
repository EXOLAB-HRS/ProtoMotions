# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 v34 cadh (server slot A, 2026-11-04 13:40 KST, user: "제안해준 안으로 얼른 학습 돌리자"): continuation of the
W1 + W2 + W3 #1 checkpoint (spdw1 s2945 e120, recipe vic4_tc_walk33_band1spd_lp15_swk_spdw1_r1) on plant human_model_v3.4
(launcher --plant v3.4: v3.3 without the gastrocnemius passive term, which made the late-swing knee 2-3x stiffer than the
human swing range) with the cadence target on the human law of the W2 judge (0.558 v + 1.060 steps/s instead of the
motion-pack 0.608 v + 1.178; alone it fixed overspeed but cost WBDS, so it is tried together with the plant fix).
Swing knee stays in r_imp (v4_imp_skip_swing_knee False, as the parent). Law, rewards, commands, distillation unchanged.
"""
import dataclasses

from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_band1spd_lp15_swk_spdw1_r1 as _base
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_band1spd_lp15_swk_spdw1_r1 import *  # noqa: F401,F403

HUMAN_CADENCE = dict(cadence_f0_slope=0.558, cadence_f0_intercept=1.060)


def env_config(robot_cfg, args):
    cfg = _base.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(cfg.control_components["steering"], **HUMAN_CADENCE)
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _base.agent_config(robot_cfg, env_cfg, args)
