# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""v4.0 all-in recipe (abd_sym_pof_ks_v40ce_r1, local run 5: symmetry 0.3, push-off band 0.5 SD, series k_eff, vic_metab 0.1,
underspeed tail 0.3) + frontal-plane head sway (2026-11-24, user: "정면에서 봤을때 좌우로 많이 흔들리는거 잡기 (티쳐)"):
lateral_sway (protomotions/envs/control/lateral_sway.py), head-over-pelvis frontal lean with a dead zone, tolerance
HC_SWAY_TOL_DEG (default 2 deg), sigma 2 deg, weight HC_SWAY_WEIGHT (default 0.3). Workshop teacher: head lateral excursion
14.4 cm vs pelvis 6.5 cm (2.2x); human head motion is not larger than the pelvis'.
"""
import dataclasses
import os

from protomotions.envs.control.lateral_sway import LATERAL_SWAY, VicSymSwaySteeringConfig, lateral_sway_reward
from protomotions.envs.mdp_component import MdpComponent

from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_up_abd_sym_pof_ks_v40ce_r1 as _all
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_up_abd_sym_pof_ks_v40ce_r1 import *  # noqa: F401,F403

SWAY_WEIGHT = float(os.environ.get("HC_SWAY_WEIGHT", "0.3"))
SWAY_TOL_DEG = float(os.environ.get("HC_SWAY_TOL_DEG", "2.0"))


def env_config(robot_cfg, args):
    cfg = _all.env_config(robot_cfg, args)
    old = cfg.control_components["steering"]
    kept = {f.name: getattr(old, f.name) for f in dataclasses.fields(old) if f.name != "_target_"}
    kept.update(sway_tol_deg=SWAY_TOL_DEG, sway_sigma_deg=2.0)
    cfg.control_components["steering"] = VicSymSwaySteeringConfig(**kept)
    cfg.reward_components["lateral_sway"] = MdpComponent(
        compute_func=lateral_sway_reward, dynamic_vars={"value": LATERAL_SWAY}, static_params={"weight": SWAY_WEIGHT})
    return cfg
