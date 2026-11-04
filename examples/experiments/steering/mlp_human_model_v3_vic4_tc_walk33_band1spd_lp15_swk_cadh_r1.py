# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 band1spd lp15 swk cadh (server slot B, 2026-11-04 10:25 KST): the first W1 + W2 + W3 pass recipe
(vic4_tc_walk33_band1spd_lp15_swk_r1) with the cadence target on the human law of the W2 judge.
Why: every run of this chain walks too fast (1.45-1.54 m/s at 1.4; 0.87 at 0.8). Turning the distillation off and a 10x
sharper steady-speed term did not change it. The chain still used the motion-pack cadence law f0 = 0.608 v + 1.178 steps/s
(2.03 steps/s at 1.4 m/s), while the human law of the judge (WBDS / Fukuchi 2018, criteria_v2.py) is 0.558 v + 1.060
(1.84 steps/s). The policy tracks the cadence target (2.02 measured), and with human-like step length it walks too fast.
vic_tc_s5_r17 made the same correction. Law, rewards, commands and distillation are unchanged.
"""
import dataclasses

from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_band1spd_lp15_swk_r1 as _base
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_band1spd_lp15_swk_r1 import *  # noqa: F401,F403


def env_config(robot_cfg, args):
    cfg = _base.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(cfg.control_components["steering"], cadence_f0_slope=0.558,
                                                             cadence_f0_intercept=1.060)
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _base.agent_config(robot_cfg, env_cfg, args)
