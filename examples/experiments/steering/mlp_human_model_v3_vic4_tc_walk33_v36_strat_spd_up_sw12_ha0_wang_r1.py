# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 v36 strat spd up sw12 ha0 wang (2026-11-08, user: "w1,2,3,2-2k 다 맞추는걸 목표로"): run with --plant v3.6.
vic4_tc_walk33_v36_strat_spd_up_sw12_ha0_r1 with the joint-angle reward (human_gait_prior) on the WBDS 2018 train5 table
(as the wang recipes) instead of Lencioni 2019.
Why: every run on the Lencioni joint-angle table walks the 1.4 m/s command at 1.15-1.29 m/s (300, 344, 352, 356, 362, 364,
366, 372 steady 2.0, 376 arm prior 0.1), while the WBDS-table runs reach it (325 1.36, 340 / 358 bias <= 0.044 m/s).
More steady-speed weight (372) and less arm prior (376) did not move it.
  human_gait_prior table_file: lencioni2019_hip_knee_ankle_by_speed_holdout5.pt -> wbds2018_hip_knee_ankle_by_speed_stand_train5.pt.
"""
from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_up_sw12_ha0_r1 as _base
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_up_sw12_ha0_r1 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_v35_strat_spd_wang_r1 import ANGLE_TABLE


def env_config(robot_cfg, args):
    cfg = _base.env_config(robot_cfg, args)
    cfg.reward_components["human_gait_prior"].static_params["table_file"] = str(ANGLE_TABLE)
    return cfg
