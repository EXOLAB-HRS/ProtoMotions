# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 v35 strat spd wang (2026-11-07, user: "Wbds로 통일하고 검증용 쪼금 빼놓는거 좋운거 같아 ㄱㄱ"):
vic4_tc_walk33_v35_strat_spd_r1 with the joint-angle reward (human_gait_prior) on WBDS 2018 instead of Lencioni 2019, so
joint angles, joint moments and GRF / CoP references all come from the same WBDS trials and the same training subjects
(id % 5 != 0; the held-out fifth, wbds2018_hip_knee_ankle_by_speed_stand_hold5.pt, is for checking that W2 is not
inflated by training on WBDS). Curves are standing-relative (minus each subject's standing angle from the HMV kit
wbds_standing_ref.csv, the W2 v4 convention; the plant's zero is its standing pose). Built with
scripts/build_human_gait_table.py --ref fukuchi --holdout-every 5 --standing-ref. At 1.0 m/s vs Lencioni: hip -3 deg
(more extension), ankle push-off -18.1 vs -21.7 deg. AMP stays on the AMASS motion pack; W2 judges are unchanged.
"""
from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_v35_strat_spd_r1 as _base
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_v35_strat_spd_r1 import *  # noqa: F401,F403

ANGLE_TABLE = _base._base.REF / "wbds2018_hip_knee_ankle_by_speed_stand_train5.pt"


def env_config(robot_cfg, args):
    cfg = _base.env_config(robot_cfg, args)
    cfg.reward_components["human_gait_prior"].static_params["table_file"] = str(ANGLE_TABLE)
    return cfg
