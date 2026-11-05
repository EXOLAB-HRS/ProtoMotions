# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 v34 tqw1 (2026-11-06): control for tqw1nd -- moment reward weight 1.0 with the feed-forward distillation kept ON
(0.5, fixed-PD hip-strategy teacher). tqw1 vs tqw1nd isolates the distillation; tqw1 vs tqp (run 239) isolates the weight.
"""
from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_v34_tqp_r1 as _tq
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_v34_tqp_r1 import *  # noqa: F401,F403

MOMENT_WEIGHT = 1.0


def env_config(robot_cfg, args):
    cfg = _tq.env_config(robot_cfg, args)
    cfg.reward_components["vic_moment_prior"].static_params["weight"] = MOMENT_WEIGHT
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _tq.agent_config(robot_cfg, env_cfg, args)
