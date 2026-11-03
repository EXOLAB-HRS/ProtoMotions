# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 band1spd swk (server slot A, 2026-11-04 03:30 KST): vic4_tc_walk33_band1spd_r1 with the swing knee back in the
impedance prior r_imp (v4_imp_skip_swing_knee False). Reason: W2 swing knee peak z -2.3 to -3.1; in swing the knee has
passive 4.6 + active 4.8 N m/rad against the human 0.6-3.9 (gain log of band1spd_r2 last at 1.4 m/s). With the swing knee
skipped the policy pays nothing for the active 4.8; back in r_imp, the passive excess (ln 4.6/3.9 = 0.17) is a small
constant and the active part gets a gradient toward 0, a looser swing knee, as in humans.
"""
import dataclasses

from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_band1spd_r1 as _base
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_band1spd_r1 import *  # noqa: F401,F403


def env_config(robot_cfg, args):
    cfg = _base.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(cfg.control_components["steering"], v4_imp_skip_swing_knee=False)
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _base.agent_config(robot_cfg, env_cfg, args)
