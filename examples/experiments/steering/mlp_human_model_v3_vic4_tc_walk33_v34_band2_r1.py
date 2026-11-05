# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 v34 band2 (2026-11-05, server slots A/B, user: "서버 두개 3안 ㄱ"): the v3.4-only W1 + W2 + W3 recipe
(vic4_tc_walk33_band1spd_lp15_swk_spdw1_r1 on --plant v3.4, parent cadence) with ONE change: v4_band_margin 1.0 -> 2.0,
so the stiffness may leave the human range (hard limit at 2x the range on each side) and only the impedance prior
(vic_imp_prior, weight 0.1, dead zone = human range) pulls it back. Question: does the policy stay inside the human range
when it is allowed out? With margin 1.0 the W3 range check passes by construction. Margin 2.0 was used before on v3.3
without lp15 / speed fixes (11-03 night: 43-68 % of knee k / b outside the band, 1.4 m/s knee collapse).
"""
import dataclasses

from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_band1spd_lp15_swk_spdw1_r1 as _base
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_band1spd_lp15_swk_spdw1_r1 import *  # noqa: F401,F403


def env_config(robot_cfg, args):
    cfg = _base.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(cfg.control_components["steering"], v4_band_margin=2.0)
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _base.agent_config(robot_cfg, env_cfg, args)
