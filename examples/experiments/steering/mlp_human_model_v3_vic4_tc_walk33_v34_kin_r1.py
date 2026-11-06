# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 v34 kin (2026-11-06, user: "판별기 = 관절 궤적, 보상 = 지면 반력·CoP + 토크" -> "1,2 는 일단 바로 시작하자"):
vic4_tc_walk33_v34_tqw1nd_armdisc_r1 (26-body motion discriminator, moment reward 1.0, distillation off, ff cap 1.0; no torque
in the discriminator) with two changes, both on the kinetics side:
- moment table -> lencioni2019_sagittal_moment_by_speed_holdout5_plantbsp.pt (Lencioni moments + the plant segment correction
  of WBDS knt_plant - knt by speed and gait percent, output/261106/e01_wbds_plant_id; swing hip / knee +0.38 / +0.21 N m/kg at
  1.0 m/s, ankle unchanged);
- + vic_kinetic_prior (weight KINETIC_WEIGHT): vertical / fore-aft ground force and CoP lever vs Lencioni 2019 force plates
  (share/human_ref/lencioni2019_grf_cop_by_speed_holdout5.pt, scripts/build_lencioni_kinetic_table.py). Basis: KAIL
  (arXiv 2603.12408) GRF + CoP terms cut hip / knee moment error 40 / 34 % at equal kinematic error; our push-off lever is
  7-9 cm vs the human 12.4 cm at 1.0 m/s. Weight 0.5 = half the moment reward (KAIL: GRF 0.25 + CoP 0.25 next to pose 0.5).
"""
import dataclasses
from pathlib import Path

from protomotions.envs.control.vic_steering import VIC_KINETIC_PRIOR, vic_gain
from protomotions.envs.mdp_component import MdpComponent

from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_v34_tqw1nd_armdisc_r1 as _base
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_v34_tqw1nd_armdisc_r1 import *  # noqa: F401,F403

KINETIC_WEIGHT = 0.5
FOOT_SENSORS = ("L_Ankle", "L_Toe", "R_Ankle", "R_Toe")   # the launcher allows exactly this robot-config difference
REF = Path(__file__).resolve().parents[4] / "share/human_ref"
MOMENT_TABLE = REF / "lencioni2019_sagittal_moment_by_speed_holdout5_plantbsp.pt"
KINETIC_TABLE = REF / "lencioni2019_grf_cop_by_speed_holdout5.pt"


def configure_robot_and_simulator(robot_cfg, simulator_cfg, args):
    """Base PD / plant setup, plus foot contact sensors: the training robot config has contact_bodies None, so the
    simulator spawns no contact sensors and vic_kinetic_prior (net foot force) would stay at its off value 1 (smoke
    2026-11-06). Sensors only read forces (filtered to the terrain); the evaluation rollouts use the same four bodies."""
    _base.configure_robot_and_simulator(robot_cfg, simulator_cfg, args)
    robot_cfg.contact_bodies = list(FOOT_SENSORS)


def env_config(robot_cfg, args):
    cfg = _base.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(
        cfg.control_components["steering"], v4_moment_table=str(MOMENT_TABLE), v4_kinetic_table=str(KINETIC_TABLE))
    cfg.reward_components["vic_kinetic_prior"] = MdpComponent(
        compute_func=vic_gain, dynamic_vars={"value": VIC_KINETIC_PRIOR}, static_params={"weight": KINETIC_WEIGHT})
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _base.agent_config(robot_cfg, env_cfg, args)
