# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 band1spd lp15 swk nodist (server slot A, 2026-11-04 07:50 KST): the first W1 + W2 + W3 pass recipe
(vic4_tc_walk33_band1spd_lp15_swk_r1) with the feed-forward distillation OFF (ff_distill weight 0).
Why: the 1.4 m/s speed sits on the W1 edge in every run of this chain, always too fast (1.449 / 1.462 / 1.472 m/s), and the
distillation teacher (v3.3 fixed-PD walker, 437d7f66) walks 0.09-0.12 m/s faster than the command at every speed
(0.891 / 1.094 / 1.513). Dropping the teacher also removes the last fixed-PD element from training (user: fixed PD out of
VIC). The law, rewards and commands are unchanged.
"""
from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_band1spd_lp15_swk_r1 as _base
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_band1spd_lp15_swk_r1 import *  # noqa: F401,F403


def env_config(robot_cfg, args):
    return _base.env_config(robot_cfg, args)


def agent_config(robot_cfg, env_cfg, args):
    cfg = _base.agent_config(robot_cfg, env_cfg, args)
    cfg.ff_distill.weight = 0.0
    return cfg
