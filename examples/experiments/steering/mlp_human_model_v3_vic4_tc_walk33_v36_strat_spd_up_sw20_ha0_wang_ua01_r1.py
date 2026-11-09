# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 v36 strat spd up sw20 ha0 wang ua01 (2026-11-09, user: "w1,2,3,2-2k 다 맞추는걸 목표로"): run with --plant v3.6.
vic4_tc_walk33_v36_strat_spd_up_sw12_ha0_wang_ua01_r1 with more weight on steady speed tracking.
Why: arm prior 0.1 (392) lets 0.8 / 1.0 m/s pass W1 at last, but the walked speed keeps drifting with training
(392 e322 -> last -> 396 last: 1.0 m/s command 0.97 -> 1.02 -> 1.23 m/s) and W2 fails once it overshoots. Steady 2.0 on
the arm-prior-0.3 recipe (390) cut the speed error and kept MoS 1.00 and W2 3/3.
  steady_speed_tracking weight 1.2 -> 2.0.
"""
from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_up_sw12_ha0_wang_ua01_r1 as _base
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_up_sw12_ha0_wang_ua01_r1 import *  # noqa: F401,F403

STEADY_WEIGHT = 2.0


def env_config(robot_cfg, args):
    cfg = _base.env_config(robot_cfg, args)
    cfg.reward_components["steady_speed_tracking"].static_params["weight"] = STEADY_WEIGHT
    return cfg
