# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 v36 strat spd up sw12 ha0 wang ua01 (2026-11-09, user: "w1,2,3,2-2k 다 맞추는걸 목표로"): run with --plant v3.6.
vic4_tc_walk33_v36_strat_spd_up_sw12_ha0_wang_r1 with less weight on the arm prior.
Why: from the same spdw1 e120 parent, run 340 (no upper-body terms) walked 0.76 / 0.99 / 1.36 m/s, while 386 (this recipe
with arm prior 0.3 and head roll 0.3) walks 0.55 / 0.78 / 1.24 m/s at e322. The arm prior is the larger of the added terms.
  vic_upper_prior weight 0.3 -> 0.1.
"""
from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_up_sw12_ha0_wang_r1 as _base
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_up_sw12_ha0_wang_r1 import *  # noqa: F401,F403

UPPER_WEIGHT = 0.1


def env_config(robot_cfg, args):
    cfg = _base.env_config(robot_cfg, args)
    cfg.reward_components["vic_upper_prior"].static_params["weight"] = UPPER_WEIGHT
    return cfg
