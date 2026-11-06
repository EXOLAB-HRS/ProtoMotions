# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 v36 strat spd wang (2026-11-07, user: "플랜트 상한은 실제 사람이 낼수 있는 최대로 반영하는게"): run with --plant v3.6
(human_model_v3.6 = v3.5 with torque ceilings raised to young-adult maximal strength: knee extension 149 -> 200, ankle
dorsi / plantar 23.7 / 118.3 -> 34 / 150, subtalar 18.2 -> 30, toe flexion 10 -> 15, wrist 9.3 / 6.7 -> 20 / 9 N m).
Everything else is vic4_tc_walk33_v35_strat_spd_wang_r1. The VIC torque normalisation (tau_res = tau_max * tanh(b), effort
and co-contraction costs, efference observation) stays at the v3.5 limits, so the parent's actions mean the same torque
and only the ceiling moves. The joint reflex cap (0.3 x effort limit) follows the new limits.
"""
import dataclasses

from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_v35_strat_spd_wang_r1 as _base
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_v35_strat_spd_wang_r1 import *  # noqa: F401,F403

# v3.5 effort limits (N m) of VicSteering.vic_joints: L/R Hip_y, Knee_y, Ankle_y, then L/R Hip_x, L/R Subtalar_x
TAU_NORM_V35 = (219.5, 149.0, 118.3, 219.5, 149.0, 118.3, 170.5, 170.5, 18.1641504178273, 18.1641504178273)


def env_config(robot_cfg, args):
    cfg = _base.env_config(robot_cfg, args)
    st = cfg.control_components["steering"]
    assert tuple(st.vic_joints) == ("L_Hip_y", "L_Knee_y", "L_Ankle_y", "R_Hip_y", "R_Knee_y", "R_Ankle_y",
                                    "L_Hip_x", "R_Hip_x", "L_Subtalar_x", "R_Subtalar_x"), st.vic_joints
    cfg.control_components["steering"] = dataclasses.replace(st, v4_tau_norm=TAU_NORM_V35)
    return cfg
