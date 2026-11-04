# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 v34 cadh kcvd (server slot B and local, 2026-11-04 16:00 KST, user: "B 쪽 재시작 ㄱㄱ"): vic4_tc_walk33_v34_cadh_kcv_r1
with the stance damping centre moved together with the stiffness centre (v4_kc_speed_damp: ln b_c += alpha_j x the ln k_c
shift; alpha hip / knee 0.5, ankle 0.15, the law's own k-b coupling). kcv alone raised k by up to x1.54 at 1.4 m/s with
b fixed, lowering the damping ratio (zeta ~ b / sqrt(k)) to about x0.81; that was not intended. Replaces the kcv run on slot B.
"""
import dataclasses

from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_v34_cadh_kcv_r1 as _base
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_v34_cadh_kcv_r1 import *  # noqa: F401,F403


def env_config(robot_cfg, args):
    cfg = _base.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(cfg.control_components["steering"], v4_kc_speed_damp=True)
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _base.agent_config(robot_cfg, env_cfg, args)
