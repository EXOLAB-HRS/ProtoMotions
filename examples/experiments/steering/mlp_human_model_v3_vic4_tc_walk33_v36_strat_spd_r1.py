# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 v36 strat spd (2026-11-07, user: "210, 350을 좀 더 학습해서 만들어볼 수 없나? ... w1,2,3,2-2k 다 맞추는걸 목표로"):
vic4_tc_walk33_v35_strat_spd_r1 (Lencioni joint-angle reward, steady speed 0.6 with tails off) run with --plant v3.6, for
continuing the two v3.5 checkpoints whose torque matched the human (run 210 last: 1.0 m/s push-off 1.49 vs 1.47, hip
-0.36 vs -0.27 N m/kg; run 300 e483: 1.38 vs 1.44, -0.33 vs -0.21) but whose speed range was compressed by the v3.5
ankle ceiling (0.8 m/s +0.07-0.11, 1.4 m/s -0.08 / -0.16 m/s). Run 325 showed v3.6 lifts that compression. The VIC torque
normalisation stays at the v3.5 limits (as in the v36 wang recipe), so the parents' actions mean the same torque.
"""
import dataclasses

from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_v35_strat_spd_r1 as _base
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_v35_strat_spd_r1 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_wang_r1 import TAU_NORM_V35


def env_config(robot_cfg, args):
    cfg = _base.env_config(robot_cfg, args)
    st = cfg.control_components["steering"]
    assert tuple(st.vic_joints) == ("L_Hip_y", "L_Knee_y", "L_Ankle_y", "R_Hip_y", "R_Knee_y", "R_Ankle_y",
                                    "L_Hip_x", "R_Hip_x", "L_Subtalar_x", "R_Subtalar_x"), st.vic_joints
    cfg.control_components["steering"] = dataclasses.replace(st, v4_tau_norm=TAU_NORM_V35)
    return cfg
