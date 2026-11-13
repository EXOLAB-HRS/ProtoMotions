# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 v36 strat spd up + arm abduction (2026-11-11, user: "팔이상한거는 고쳐서 워크샵 들고가는게 나을거 같은데, 이거 고쳐서
티쳐 만드는거 344에서 시작해서"): teacher 344 last (= this recipe's parent, vic4_tc_walk33_v36_strat_spd_up_r1) holds the upper
arms 38-45 deg away from the body; Lencioni 2019 adults 16.4 +- 3.8 deg (854 walking trials, 15-17 deg over speed). Neither
vic_upper_prior (sagittal + elbow) nor W2-3 looked at the frontal plane. Adds ONE task term, everything else unchanged:
- vic_arm_abd_prior (ABD_WEIGHT): exp(-0.5 mean_arms z^2), z = (abduction - 16.4 deg) / (3 x 3.8 deg).
"""
import dataclasses
import math

from protomotions.envs.control.vic_steering import VIC_ARM_ABD_PRIOR, vic_gain
from protomotions.envs.mdp_component import MdpComponent

from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_up_r1 as _base
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_up_r1 import *  # noqa: F401,F403

ABD_MEAN_DEG = 16.4
ABD_SD_DEG = 3.8
# 2026-11-12 (hc_root_armres only): HC_ABD_WEIGHT / HC_ABD_KERNEL override for the residual arm runs (r1 used 0.3 / 3.0 and the
# abduction barely moved, 37/50 -> 36/46 deg); unset = the abd r1 values
import os
ABD_KERNEL = float(os.environ.get("HC_ABD_KERNEL", "3.0"))
ABD_WEIGHT = float(os.environ.get("HC_ABD_WEIGHT", "0.3"))


def env_config(robot_cfg, args):
    cfg = _base.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(
        cfg.control_components["steering"], v4_arm_abd=(math.radians(ABD_MEAN_DEG), math.radians(ABD_SD_DEG), ABD_KERNEL))
    cfg.reward_components["vic_arm_abd_prior"] = MdpComponent(
        compute_func=vic_gain, dynamic_vars={"value": VIC_ARM_ABD_PRIOR}, static_params={"weight": ABD_WEIGHT})
    return cfg
