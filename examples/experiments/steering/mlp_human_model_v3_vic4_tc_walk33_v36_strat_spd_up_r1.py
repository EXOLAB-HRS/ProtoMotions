# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 v36 strat spd up (2026-11-07, user: "팔이랑 머리는 왜 저렇게 이상하게 움직이지? 많이 흔들리지 않아? ... 다음 학습부터는
이것도 잘 맞추게", "머리, 팔쪽도 고려해서 ㄱㄱ"): vic4_tc_walk33_v36_strat_spd_r1 plus two upper-body task-reward terms (no
training term looked at the arms or the head pitch; W2-3 at 1.0 m/s on runs 210 / 300 / 325 / 340: arm swing range 52-56
deg vs human 18 +- 7, elbow mean 19-23 vs 31 +- 5 deg, arm-leg r 0.6-0.9 vs 0.89, head pitch range 4-15 vs 4.6 +- 2.7 deg):
- vic_upper_prior (UPPER_WEIGHT): both arms' sagittal angle and elbow angle against the Lencioni 2019 train5 profile at each
  leg's gait percent and the commanded speed (share/human_ref/lencioni2019_arm_by_speed_train5.pt, W2-3 definitions);
- head_angular_stability (HEAD_WEIGHT): exp(-|w_pitch,roll|^2 / HEAD_RATE^2) of the head body, world pitch / roll rate
  (an existing kernel, locomotion_quality.py); head_roll_stability stays.
"""
import dataclasses
from pathlib import Path

from protomotions.envs.context_views import EnvContext
from protomotions.envs.control.vic_steering import VIC_UPPER_PRIOR, vic_gain
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.rewards.locomotion_quality import head_angular_stability

from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_r1 as _base
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_r1 import *  # noqa: F401,F403

UPPER_TABLE = Path(__file__).resolve().parents[4] / "share/human_ref/lencioni2019_arm_by_speed_train5.pt"
UPPER_WEIGHT = 0.3
HEAD_WEIGHT = 0.15
HEAD_RATE = 0.5      # rad/s; human head pitch range 4.6 deg per ~1.1 s stride ~ 0.2 rad/s peak, the runs ~2-3x that


def add_upper(cfg, robot_cfg):
    assert UPPER_TABLE.is_file(), UPPER_TABLE
    cfg.control_components["steering"] = dataclasses.replace(cfg.control_components["steering"], v4_upper_table=str(UPPER_TABLE))
    cfg.reward_components["vic_upper_prior"] = MdpComponent(
        compute_func=vic_gain, dynamic_vars={"value": VIC_UPPER_PRIOR}, static_params={"weight": UPPER_WEIGHT})
    cfg.reward_components["head_angular_stability"] = MdpComponent(
        compute_func=head_angular_stability, dynamic_vars={"body_ang_vel": EnvContext.current.rigid_body_ang_vel},
        static_params={"body_index": list(robot_cfg.kinematic_info.body_names).index("Head"), "speed_scale": HEAD_RATE,
                       "weight": HEAD_WEIGHT})
    return cfg


def env_config(robot_cfg, args):
    return add_upper(_base.env_config(robot_cfg, args), robot_cfg)
