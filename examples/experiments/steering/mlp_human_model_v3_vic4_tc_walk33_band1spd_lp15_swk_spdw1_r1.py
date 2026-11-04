# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 band1spd lp15 swk spdw1 (server slot B, 2026-11-04 09:10 KST): the first W1 + W2 + W3 pass recipe
(vic4_tc_walk33_band1spd_lp15_swk_r1) with the steady-speed term's tolerance matched to the W1 speed gate.
Why: every run of the chain walks too fast at 1.4 m/s (1.449-1.538 m/s; turning the distillation off made it worse, so the
teacher is not the cause). steady_speed_tracking was exp(-20 e^2) (effective sigma 1/sqrt(40) = 0.16 m/s, 3x the W1 gate
0.05 m/s and 4.5x the human within-trial speed SD 0.035 m/s, Collins & Kuo 2013), and its overspeed tail (scale 8, weight 0.5)
made small overspeed cheaper still. Here the steady term (scored only outside speed transitions) uses sigma = 0.05 m/s,
scale 1 / (2 sigma^2) = 200, the same on both sides, no tails. Weight, transition term, law and commands unchanged.
"""
from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_band1spd_lp15_swk_r1 as _base
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_band1spd_lp15_swk_r1 import *  # noqa: F401,F403

STEADY_SCALE = 200.0


def env_config(robot_cfg, args):
    cfg = _base.env_config(robot_cfg, args)
    sp = cfg.reward_components["steady_speed_tracking"].static_params
    sp.update(vel_err_scale=STEADY_SCALE, overspeed_err_scale=STEADY_SCALE, overspeed_tail_weight=0.0, underspeed_tail_weight=0.0)
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _base.agent_config(robot_cfg, env_cfg, args)
