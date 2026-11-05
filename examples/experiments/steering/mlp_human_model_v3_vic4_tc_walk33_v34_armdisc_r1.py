# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 v34 armdisc (2026-11-05, user: "팔이 이상하게 움직이던디 좀 예쁘게 ... 머리 흔들림 이런것도"): the v3.4-only W1 + W2 + W3
recipe (vic4_tc_walk33_band1spd_lp15_swk_spdw1_r1 on --plant v3.4, parent cadence) with ONE change: the AMP discriminator
(disc_coords_obs, 2 history steps) sees all 26 bodies (neck, head, thorax, shoulders, elbows, wrists, hands added) instead of
the 14 pelvis / leg / trunk bodies, so arm swing and head motion are pulled toward the motion pack. The launcher widens
the parent's discriminator with zero-weight columns (expand_disc_inputs legs -> full), so it starts equal to the parent.
"""
from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_band1spd_lp15_swk_spdw1_r1 as _base
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_band1spd_lp15_swk_spdw1_r1 import *  # noqa: F401,F403

DISC_KEY = "disc_coords_obs"
FULL_BODIES = list(range(26))


def env_config(robot_cfg, args):
    cfg = _base.env_config(robot_cfg, args)
    cfg.observation_components[DISC_KEY].static_params["body_ids"] = FULL_BODIES
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    cfg = _base.agent_config(robot_cfg, env_cfg, args)
    cfg.reference_obs_components[DISC_KEY].static_params["body_ids"] = FULL_BODIES
    return cfg
