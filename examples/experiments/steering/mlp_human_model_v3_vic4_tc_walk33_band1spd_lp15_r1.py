# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 band1spd lp15 (server slot C, one night only, 2026-11-04 KST): vic4_tc_walk33_band1spd_r1 with the feed-forward
activation lowpass 15 ms instead of 30 ms (tau_ff_lowpass_s). Reason: the 11-03 tau_ff diagnostic found only 53-63 % of
the fast knee torque coverable by the 30 Hz feed-forward behind a 30 ms lowpass (>= 90 % at hip / ankle); that gap is what
knee beta 0.2 (hidden fixed PD) filled. 15 ms stays inside muscle activation time constants (activation about 10-20 ms,
deactivation about 40 ms; Zajac 1989, Thelen 2003), so the fast knee torque comes from feed-forward, not hidden PD.
"""
import dataclasses

from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_band1spd_r1 as _base
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_band1spd_r1 import *  # noqa: F401,F403


def env_config(robot_cfg, args):
    cfg = _base.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(cfg.control_components["steering"], tau_ff_lowpass_s=0.015)
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _base.agent_config(robot_cfg, env_cfg, args)
