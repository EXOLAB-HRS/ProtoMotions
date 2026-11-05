# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 v34 kcvd preact (2026-11-05, server, user: "속도에는 좀 반응하게, heel strike 탁 할때 강성 탁 오르는거 처럼"):
vic4_tc_walk33_v34_kcvd_r1 (v3.4-only recipe + speed-dependent stance k, b centres, parent cadence; slot A 11-05 night:
W1 3/3 seeds, W2 1/3) plus terminal-swing pre-activation of the hip and knee (v4_preact_joints, ramp over 80-100 % of the
swing). In the v3.4 runs the knee stiffness rises 2 -> 38 N m/rad one control step AFTER the load is seen; people
pre-activate hamstrings and vasti 50-100 ms BEFORE contact. Candidate options, no version bump until approved.
"""
import dataclasses

from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_v34_kcvd_r1 as _base
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_v34_kcvd_r1 import *  # noqa: F401,F403


def env_config(robot_cfg, args):
    cfg = _base.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(cfg.control_components["steering"],
                                                             v4_preact_joints=("Hip_y", "Knee_y"), v4_preact_p=(0.8, 1.0))
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _base.agent_config(robot_cfg, env_cfg, args)
