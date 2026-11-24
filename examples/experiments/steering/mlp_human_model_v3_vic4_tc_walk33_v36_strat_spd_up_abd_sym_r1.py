# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 v36 strat spd up abd + left-right symmetry (2026-11-14, user: "하체라도 비대칭이 없게 예쁘게 걸어야 ...
잔차학습으로 비대칭 문제 해결해보자, 어차피 티쳐는 재료니까 누더기여도 되는"). Meant for the residual composition of teacher r4
(actor mean = mu_344 + 0.3 (mu - mu_344), Shoulder_x locked, arm shift 38,62; hc_root_armres env hooks), continued from r4 last.

Audit 11-14 (1.0 m/s, seed 7, 7-15 s): r4 ROM R/L ankle 1.47, hip ab/adduction 1.33, subtalar 1.23, mean hip rotation L -5 / R -10.5 deg,
stance 0.60 / 0.62; 344 knee ROM 0.85, stance 0.58 / 0.63; students copy it (7484 ankle 1.48).

Adds ONE task term to the abd r1 recipe (r4 ran it with HC_ABD_WEIGHT=0): gait_symmetry (gait_symmetry.VicSymSteering) with the
human dead zone and the stance-time factor, weight HC_SYM_WEIGHT (default 0.3; vic3 r1 used 0.2 with the dead zone, r8 0.2 without
it hurt the leg curves when the whole policy moved; here the residual weight 0.3 bounds how far the gait can move from 344).
Joints: the six leg DOFs (the 10-28 set had only the sagittal three; the 11-14 audit asymmetry is mostly frontal / transverse) and
the arm joints the residual can move (Shoulder_x is locked to 344 + shift, so it is left out).
Dead zones not in SYM_TOL_DEG (no L/R human data in WBDS / Van Criekinge for these, assumption): Hip_x 4, Hip_z 5, Subtalar_x 5 deg.
Mirror signs: Hip_x, Hip_z, Subtalar_x, Shoulder_z, Elbow_z mirrored (model axes, L/R means of 344 opposite in sign).
"""
import dataclasses
import os

from protomotions.envs.control.gait_symmetry import GAIT_SYMMETRY, VicSymSteeringConfig, gait_symmetry_reward
from protomotions.envs.mdp_component import MdpComponent

from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_up_abd_r1 as _abd
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_up_abd_r1 import *  # noqa: F401,F403

SYM_WEIGHT = float(os.environ.get("HC_SYM_WEIGHT", "0.3"))
SYM_JOINTS = (("Hip_y", 1.0, 5.0), ("Hip_x", -1.0, 4.0), ("Hip_z", -1.0, 5.0), ("Knee_y", 1.0, 5.0), ("Ankle_y", 1.0, 5.0),
              ("Subtalar_x", -1.0, 5.0), ("Shoulder_y", 1.0, 6.0), ("Shoulder_z", -1.0, 6.0), ("Elbow_z", -1.0, 6.0))
SYM_TOL_EXTRA = {"Hip_x": 4.0, "Hip_z": 5.0, "Subtalar_x": 5.0}
SYM = dict(sym_joints=SYM_JOINTS, sym_deadzone=True, sym_tol_deg=SYM_TOL_EXTRA, sym_stance_tol_pct=3.73, sym_stance_sigma_pct=2.0)


def env_config(robot_cfg, args):
    cfg = _abd.env_config(robot_cfg, args)
    old = cfg.control_components["steering"]
    kept = {f.name: getattr(old, f.name) for f in dataclasses.fields(old) if f.name != "_target_"}
    kept.update(SYM)
    cfg.control_components["steering"] = VicSymSteeringConfig(**kept)
    cfg.reward_components["gait_symmetry"] = MdpComponent(
        compute_func=gait_symmetry_reward, dynamic_vars={"value": GAIT_SYMMETRY}, static_params={"weight": SYM_WEIGHT})
    return cfg
