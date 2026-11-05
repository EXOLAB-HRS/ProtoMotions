# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 v34 tqw1nd armdisc (2026-11-06): tqw1nd (moment reward 1.0, distillation off, ff cap 1.0) plus the arm / head
discriminator of vic4_tc_walk33_v34_armdisc_r1 (disc_coords_obs over all 26 bodies; launcher widens the parent disc).
"""
from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_v34_tqw1nd_r1 as _tq
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_v34_tqw1nd_r1 import *  # noqa: F401,F403

DISC_KEY = "disc_coords_obs"
FULL_BODIES = list(range(26))


def env_config(robot_cfg, args):
    cfg = _tq.env_config(robot_cfg, args)
    cfg.observation_components[DISC_KEY].static_params["body_ids"] = FULL_BODIES
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    cfg = _tq.agent_config(robot_cfg, env_cfg, args)
    cfg.reference_obs_components[DISC_KEY].static_params["body_ids"] = FULL_BODIES
    return cfg
