# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 v36 strat spd up abd r1 + v4_metab_ce (2026-11-15, human-controller local v4.0 residual teacher on 344): the abd r1
recipe unchanged (run with HC_ABD_WEIGHT=0) and the metabolic proxy's ankle power taken from the series elastic element's
contractile element (VicSteering.v4_metab_ce), so Achilles recoil is not counted as muscle work. Plant human_model_v4.0."""
import dataclasses

from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_up_abd_r1 as _base
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_up_abd_r1 import *  # noqa: F401,F403


def env_config(robot_cfg, args):
    cfg = _base.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(cfg.control_components["steering"], v4_metab_ce=True)
    return cfg
