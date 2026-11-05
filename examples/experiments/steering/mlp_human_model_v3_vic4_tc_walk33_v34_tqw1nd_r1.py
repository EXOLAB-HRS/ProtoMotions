# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 v34 tqw1nd (2026-11-06, user: "토크검증 통과하는 방향으로"): vic4_tc_walk33_v34_tqp_r1 (ff100 + human joint-moment
profile reward) with TWO coupled changes, both aimed at the hip-strategy gait:
- moment reward weight 0.25 -> 1.0 (in run 239 its raw value fell 0.87 -> 0.55 while the other terms rose: 0.25 of a ~2.6
  task sum was not enough to hold the torque shape);
- feed-forward distillation OFF (ff_distill weight 0.5 -> 0): the distillation teacher (v3.3 fixed-PD walker 437d7f66) is
  itself hip-strategy (W2-2 at 1.0 m/s: late-stance hip -2.10 vs human -0.28 N m/kg, push-off 1.08 vs 1.51), so it pulls
  tau_ff toward the torque pattern the reward is trying to remove.
"""
from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_v34_tqp_r1 as _tq
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_v34_tqp_r1 import *  # noqa: F401,F403

MOMENT_WEIGHT = 1.0
FFD_WEIGHT = 0.0


def env_config(robot_cfg, args):
    cfg = _tq.env_config(robot_cfg, args)
    cfg.reward_components["vic_moment_prior"].static_params["weight"] = MOMENT_WEIGHT
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    cfg = _tq.agent_config(robot_cfg, env_cfg, args)
    cfg.ff_distill.weight = FFD_WEIGHT
    return cfg
