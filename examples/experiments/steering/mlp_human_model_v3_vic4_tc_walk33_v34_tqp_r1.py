# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 v34 tqp (2026-11-05, user: "토크 맞출수 있게 보상 조정을 하던지 해서 학습해보자"): vic4_tc_walk33_v34_ff100_r1 (v3.4-only
recipe with the feed-forward cap at the joint strength, tau_res_scale 1.0) plus the human joint-moment profile reward
vic_moment_prior (weight MOMENT_WEIGHT, as human_gait_prior's 0.25 for angles). Table: Lencioni et al. 2019 sagittal hip /
knee / ankle moments by speed and gait percent, adults, subjects id % 5 == 0 held out (the same split as the angle table);
the W2-2 judge uses WBDS 2018, a different dataset. Sign map table -> plant (+ hip extension, knee flexion, ankle
plantarflexion): hip +1, knee -1 (Lencioni knee moments are extension-positive), ankle +1, checked against WBDS
(full table vs WBDS at 1.0 m/s: r 0.75 / 0.51 / 0.99; 475 strides, 24 training subjects). Kernel: z = (tau - mean) / (1.5 max(sd, 0.1)).
"""
import dataclasses
from pathlib import Path

from protomotions.envs.control.vic_steering import VIC_MOMENT_PRIOR, vic_gain
from protomotions.envs.mdp_component import MdpComponent

from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_v34_ff100_r1 as _base
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_v34_ff100_r1 import *  # noqa: F401,F403

MOMENT_WEIGHT = 0.25
MOMENT_TABLE = Path(__file__).resolve().parents[4] / "share/human_ref/lencioni2019_sagittal_moment_by_speed_holdout5.pt"


def env_config(robot_cfg, args):
    cfg = _base.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(
        cfg.control_components["steering"], v4_moment_table=str(MOMENT_TABLE), v4_moment_signs=(1.0, -1.0, 1.0),
        v4_moment_sd_floor=0.1, v4_moment_kernel=1.5)
    cfg.reward_components["vic_moment_prior"] = MdpComponent(
        compute_func=vic_gain, dynamic_vars={"value": VIC_MOMENT_PRIOR}, static_params={"weight": MOMENT_WEIGHT})
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _base.agent_config(robot_cfg, env_cfg, args)
