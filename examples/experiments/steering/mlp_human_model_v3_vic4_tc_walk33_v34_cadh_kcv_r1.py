# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 v34 cadh kcv (server slot B and local, 2026-11-04 13:40 KST): vic4_tc_walk33_v34_cadh_r1 plus the speed-dependent
stance reference centre (v4_kc_speed_gamma, user-approved "속도별 강성 기준"). In the #1 checkpoint the stance k_eff rose only
4-5 % from 0.8 to 1.4 m/s because k_c did not depend on speed and the policy learned a near-constant stiffness offset
(a SD 0.08-0.13). Human stance quasi-stiffness grows as v^gamma (WBDS / Fukuchi 2018 sagittal moment vs angle, 0.6-1.6 m/s:
hip 0.64 (angle from Lencioni 2019), knee 0.87, ankle 0.52; x1.37-1.54 from 0.8 to 1.4 m/s); only this ratio is used.
Range width, swing / standing centres and the W3 judge are unchanged.
"""
import dataclasses

from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_v34_cadh_r1 as _base
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_v34_cadh_r1 import *  # noqa: F401,F403

KC_SPEED_GAMMA = {"Hip_y": 0.64, "Knee_y": 0.87, "Ankle_y": 0.52}


def env_config(robot_cfg, args):
    cfg = _base.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(cfg.control_components["steering"],
                                                             v4_kc_speed_gamma=dict(KC_SPEED_GAMMA))
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _base.agent_config(robot_cfg, env_cfg, args)
