# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 v34 qlp3 (2026-11-05, local, user: "로컬 1안 ㄱㄱ"; candidate option, no version bump until approved): the v3.4-only
W1 + W2 + W3 recipe (vic4_tc_walk33_band1spd_lp15_swk_spdw1_r1 on --plant v3.4, parent cadence) with ONE change:
v4_qref_lowpass_hz 3.0, a first-order lowpass on the leg PD targets q_ref. q_ref keeps the slow intent; fast deviations
(heel strike, push-off) are answered by k and tau_ff, so stiffness is no longer interchangeable with q_ref on fast time
scales. Band margin 1.0 as the base. Watch: stiffness use (a away from 0 by phase), 1.4 m/s knee, q_ref action jumps.
"""
import dataclasses

from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_band1spd_lp15_swk_spdw1_r1 as _base
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_band1spd_lp15_swk_spdw1_r1 import *  # noqa: F401,F403


def env_config(robot_cfg, args):
    cfg = _base.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(cfg.control_components["steering"], v4_qref_lowpass_hz=3.0)
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _base.agent_config(robot_cfg, env_cfg, args)
