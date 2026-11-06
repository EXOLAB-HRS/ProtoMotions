# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 v35 strat (2026-11-07, user: "플랜트 질량, 관성모멘트 ... 새버젼 해서 업뎃해줘", "검증 데이터도 ... 한번에 실험한걸로"):
vic4_tc_walk33_v34_strat_r1 for plant human_model_v3.5 (run with --plant v3.5: de Leva segment masses / centres of mass /
inertias, leg inertia about the hip 2.88 -> 1.85 kg m^2), with both kinetic tables from the same WBDS 2018 trials and the same
training subjects (id % 5 != 0; the held-out fifth is judged separately):
- moments: wbds2018_sagittal_moment_by_speed_plantbsp_v35_train5.pt (Visual3D moments + v3.5 - Dempster segment terms,
  scripts/wbds_sagittal_id.py --tag _v35; plant signs);
- ground force / CoP: wbds2018_grf_cop_by_speed_train5.pt (the force plates of those strides; at 1.0 m/s push-off
  fx 0.155 BW, lever 0.078 body heights, the same as Lencioni 0.166 / 0.077).
"""
import dataclasses

from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_v34_strat_r1 as _base
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_v34_strat_r1 import *  # noqa: F401,F403

MOMENT_TABLE = _base.REF / "wbds2018_sagittal_moment_by_speed_plantbsp_v35_train5.pt"
KINETIC_TABLE = _base.REF / "wbds2018_grf_cop_by_speed_train5.pt"


def env_config(robot_cfg, args):
    cfg = _base.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(
        cfg.control_components["steering"], v4_moment_table=str(MOMENT_TABLE), v4_kinetic_table=str(KINETIC_TABLE))
    return cfg
