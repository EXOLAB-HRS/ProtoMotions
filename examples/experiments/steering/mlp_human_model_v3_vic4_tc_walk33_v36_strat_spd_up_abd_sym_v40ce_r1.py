# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 v36 strat spd up abd + v4_metab_ce + left-right symmetry (2026-11-15, human-controller local v4.0 residual teacher,
user: "걍 전부 다 넣고 하지"): the abd_v40ce_r1 recipe (plant human_model_v4.0, metabolic proxy from the ankle CE) plus the
gait_symmetry term of the server armsym1 recipe (abd_sym_r1, 11-14) unchanged: six leg DOFs + Shoulder_y / Shoulder_z /
Elbow_z, human dead zone (Hip_x 4, Hip_z 5, Subtalar_x 5 deg assumed; others SYM_TOL_DEG), stance-time factor 3.73 %p,
weight HC_SYM_WEIGHT (default 0.3)."""
import dataclasses
import os

from protomotions.envs.control.gait_symmetry import GAIT_SYMMETRY, VicSymSteeringConfig, gait_symmetry_reward
from protomotions.envs.mdp_component import MdpComponent

from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_up_abd_v40ce_r1 as _v40
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_up_abd_v40ce_r1 import *  # noqa: F401,F403

SYM_WEIGHT = float(os.environ.get("HC_SYM_WEIGHT", "0.3"))
SYM_JOINTS = (("Hip_y", 1.0, 5.0), ("Hip_x", -1.0, 4.0), ("Hip_z", -1.0, 5.0), ("Knee_y", 1.0, 5.0), ("Ankle_y", 1.0, 5.0),
              ("Subtalar_x", -1.0, 5.0), ("Shoulder_y", 1.0, 6.0), ("Shoulder_z", -1.0, 6.0), ("Elbow_z", -1.0, 6.0))
SYM_TOL_EXTRA = {"Hip_x": 4.0, "Hip_z": 5.0, "Subtalar_x": 5.0}
SYM = dict(sym_joints=SYM_JOINTS, sym_deadzone=True, sym_tol_deg=SYM_TOL_EXTRA, sym_stance_tol_pct=3.73, sym_stance_sigma_pct=2.0)


def env_config(robot_cfg, args):
    cfg = _v40.env_config(robot_cfg, args)
    old = cfg.control_components["steering"]
    kept = {f.name: getattr(old, f.name) for f in dataclasses.fields(old) if f.name != "_target_"}
    kept.update(SYM)
    cfg.control_components["steering"] = VicSymSteeringConfig(**kept)
    cfg.reward_components["gait_symmetry"] = MdpComponent(
        compute_func=gait_symmetry_reward, dynamic_vars={"value": GAIT_SYMMETRY}, static_params={"weight": SYM_WEIGHT})
    return cfg
