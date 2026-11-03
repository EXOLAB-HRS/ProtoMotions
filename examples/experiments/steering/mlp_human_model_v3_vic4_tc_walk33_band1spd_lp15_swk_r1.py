# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 band1spd lp15 swk (server slot B, 2026-11-04 04:20 KST): the two physiological knee fixes together on top of
vic4_tc_walk33_band1spd_r1: feed-forward lowpass 15 ms (lp15 rescued stiffness opening from the walker, 1.4 knee r 0.976)
and the swing knee back in r_imp (swing knee passive 4.6 + active 4.8 N m/rad vs human 0.6-3.9). Target: W2 swing knee
peak (z -2.6 to -3.1) and the last WBDS rows of band1spd_r2b (W1 + W3 pass, WBDS 0.688 vs 0.711).
"""
import dataclasses

from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_band1spd_r1 as _base
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_band1spd_r1 import *  # noqa: F401,F403


def env_config(robot_cfg, args):
    cfg = _base.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(cfg.control_components["steering"], tau_ff_lowpass_s=0.015,
                                                             v4_imp_skip_swing_knee=False)
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _base.agent_config(robot_cfg, env_cfg, args)
