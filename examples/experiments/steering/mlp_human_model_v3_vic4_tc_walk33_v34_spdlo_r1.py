# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 v34 spdlo (2026-11-05 local night, user: "ㄱㄱ"): the v3.4-only W1 + W2 + W3 recipe
(vic4_tc_walk33_band1spd_lp15_swk_spdw1_r1 on --plant v3.4, parent cadence, NO speed-dependent stiffness) with ONE change:
the steady-speed tolerance widened from sigma 0.05 m/s (spdw1, scale 200) to sigma 0.16 m/s (scale 20), symmetric, no tails.
Why: the server froze the stiffness structure (11-05 20:30, no kcvd in the teacher); kcvd-spdlo s2946 e322 reached W2 v5
0.771/0.741 while failing W1 only by 0.8 m/s overspeed 0.062, so this checks whether the speed relaxation alone moves W2.
Pair: v3.4-only s2946 (spdw1), same seed.
"""
from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_band1spd_lp15_swk_spdw1_r1 as _base
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_band1spd_lp15_swk_spdw1_r1 import *  # noqa: F401,F403

STEADY_SCALE = 20.0


def env_config(robot_cfg, args):
    cfg = _base.env_config(robot_cfg, args)
    sp = cfg.reward_components["steady_speed_tracking"].static_params
    sp.update(vel_err_scale=STEADY_SCALE, overspeed_err_scale=STEADY_SCALE, overspeed_tail_weight=0.0, underspeed_tail_weight=0.0)
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _base.agent_config(robot_cfg, env_cfg, args)
