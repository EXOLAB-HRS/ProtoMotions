# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 v34 strat (2026-11-07, user: "w1, w2, w3는 기존처럼 잘 통과하면서 ... push off 많이 쓰고, 엉덩이 덜쓰는 쪽으로 ... 보상으로
만들어야할거 같은데"): vic4_tc_walk33_v34_kin_r1 with three changes.
- moment table -> wbds2018_sagittal_moment_by_speed_plantbsp_train5.pt (WBDS 2018, subjects id % 5 != 0, Visual3D moments
  refit to the plant segments; already in plant signs). Lencioni (the kin table) has push-off 1.26 / hip min -0.56 N m/kg at
  1.0 m/s against WBDS 1.48 / -0.35, so the reward aimed below the judge; now training and judging use one lab's protocol.
- + vic_strategy_cost (weight STRATEGY_WEIGHT): band cost on hip (two-sided) and ankle (below the band only) moments; the
  kernel moment reward saturated (kin last: hip -0.99, push-off 1.07 N m/kg at 1.0 m/s).
- steady_speed_tracking weight 0.15 -> STEADY_WEIGHT with its overspeed tail on: kin overspeeds +0.06-0.16 m/s (W1 fail).
- v4_kinetic_friction: the fore-aft ground force adds the PhysX friction force (the contact sensor net force is the normal
  force only, so fx was 0 in the kin run; rollout check: braking -0.08, propulsion +0.08-0.10 BW, human -0.11 / +0.16).
"""
import dataclasses
from pathlib import Path

from protomotions.envs.control.vic_steering import VIC_STRATEGY_COST, vic_cost
from protomotions.envs.mdp_component import MdpComponent

from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_v34_kin_r1 as _base
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_v34_kin_r1 import *  # noqa: F401,F403

STRATEGY_WEIGHT = 0.3
STEADY_WEIGHT = 0.4
STEADY_OVERSPEED_TAIL = 0.5
# band cost = (distance outside the band / STRATEGY_SCALE)^2, capped at STRATEGY_CAP: 0.3 N m/kg and 9 keep a gradient up to
# 0.9 N m/kg outside the band (smoke with 0.1 / 4: the cost sat at the cap, kin last hip is 0.55 N m/kg below the band).
STRATEGY_SCALE = 0.3
STRATEGY_CAP = 9.0
MOMENT_TABLE = _base.REF / "wbds2018_sagittal_moment_by_speed_plantbsp_train5.pt"


def env_config(robot_cfg, args):
    cfg = _base.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(
        cfg.control_components["steering"], v4_moment_table=str(MOMENT_TABLE), v4_moment_signs=(1.0, 1.0, 1.0),
        v4_strategy=True, v4_strategy_scale=STRATEGY_SCALE, v4_strategy_cap=STRATEGY_CAP, v4_kinetic_friction=True)
    cfg.reward_components["vic_strategy_cost"] = MdpComponent(
        compute_func=vic_cost, dynamic_vars={"value": VIC_STRATEGY_COST}, static_params={"weight": STRATEGY_WEIGHT})
    sp = cfg.reward_components["steady_speed_tracking"].static_params
    sp["weight"] = STEADY_WEIGHT
    sp["overspeed_tail_weight"] = STEADY_OVERSPEED_TAIL
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _base.agent_config(robot_cfg, env_cfg, args)
