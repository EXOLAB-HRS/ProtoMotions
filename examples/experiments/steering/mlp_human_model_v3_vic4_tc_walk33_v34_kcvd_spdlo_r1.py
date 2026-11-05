# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 v34 kcvd spdlo (2026-11-05 local night, user: "속도 추종은 좀 떨어지더라도 속도별 강성·관절궤적을 더 맞추자"): the kcvd
recipe (v3.4, speed-dependent stance stiffness and damping centres, parent cadence) with the steady-speed tolerance widened
back from sigma 0.05 m/s (spdw1, scale 200) to sigma 0.16 m/s (scale 20, the pre-spdw1 value), kept symmetric with no tails.
Why: both kcvd seeds gave human-like stance stiffness speed ratios (Hip_y 1.43, Ankle_y 1.35) but failed only W1 by overspeed
0.04-0.11 m/s; the tight steady term is the one place the policy trades gait shape for speed precision.
"""
from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_v34_kcvd_r1 as _base
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_v34_kcvd_r1 import *  # noqa: F401,F403

STEADY_SCALE = 20.0


def env_config(robot_cfg, args):
    cfg = _base.env_config(robot_cfg, args)
    sp = cfg.reward_components["steady_speed_tracking"].static_params
    sp.update(vel_err_scale=STEADY_SCALE, overspeed_err_scale=STEADY_SCALE, overspeed_tail_weight=0.0, underspeed_tail_weight=0.0)
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _base.agent_config(robot_cfg, env_cfg, args)
