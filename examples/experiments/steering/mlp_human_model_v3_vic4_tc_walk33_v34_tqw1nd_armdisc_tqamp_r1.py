# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 v34 tqw1nd armdisc tqamp (2026-11-06, user: "방법1 로 학습 바로 돌리자", "상체도 같이"): server 247's recipe
(vic4_tc_walk33_v34_tqw1nd_armdisc_r1: moment reward 1.0, distillation off, ff cap 1.0, 26-body discriminator; 247 s2944
passed W1 + W2 + W3 at all rollout seeds with torque near ff100) with ONE change: the AMP discriminator also sees the leg
joint moments (key disc_moment_obs, 6 columns appended after the parent's inputs; agent = plant torque in N m/kg, expert =
Lencioni 2019 moment of the reference frame's gait percent and speed + between-subject noise,
protomotions/envs/obs/disc_moment.py). The launcher appends zero-weight columns with a unit normalizer, so the
discriminator starts equal to the parent's.
"""
import dataclasses

from protomotions.envs.control.vic_steering import VIC_MOMENT_FEAT
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.obs.disc_moment import compute_disc_moment_from_motion_lib, compute_disc_moment_obs

from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_v34_tqw1nd_armdisc_r1 as _ad
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_v34_tqw1nd_armdisc_r1 import *  # noqa: F401,F403

MOMENT_KEY = "disc_moment_obs"
MOMENT_NOISE = 1.0            # expert sample = table mean + MOMENT_NOISE * sd * N(0, 1) per joint


def env_config(robot_cfg, args):
    cfg = _ad.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(cfg.control_components["steering"], v4_moment_disc=True)
    cfg.observation_components[MOMENT_KEY] = MdpComponent(
        compute_func=compute_disc_moment_obs, dynamic_vars={"vic_moment_feat": VIC_MOMENT_FEAT})
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    cfg = _ad.agent_config(robot_cfg, env_cfg, args)
    m = cfg.model
    # Append last so the parent's discriminator columns keep their positions.
    for module in (m.discriminator, m.discriminator.models[0]):
        module.in_keys = list(module.in_keys) + [MOMENT_KEY]
    if MOMENT_KEY not in m.in_keys:
        m.in_keys = list(m.in_keys) + [MOMENT_KEY]
    st = env_cfg.control_components["steering"]
    cfg.reference_obs_components[MOMENT_KEY] = MdpComponent(
        compute_func=compute_disc_moment_from_motion_lib, dynamic_vars={},
        static_params={"table_file": st.v4_moment_table, "signs": tuple(st.v4_moment_signs),
                       "feet": [[4, 5], [9, 10]], "stance_heights": tuple(st.stance_heights),
                       "noise": MOMENT_NOISE, "sd_floor": st.v4_moment_sd_floor})
    return cfg
