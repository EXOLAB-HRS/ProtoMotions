# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""vic4c_TC_s5_from-s5_noeff_r1 (stage 1 of the low-stiffness path, local, 2026-11-02; user: "s5 → 저강성 즉시 전환,
80M"): the s5 A2 recipe without VIC effort (vic4_TC_s5_from-s5_noeff_r1) with the stiffness outputs frozen at the human
impedance centre from the first step (v4_freeze_a, a = 0: k_tot = k_c(phi), b_tot = b_c(phi)), alpha 1 / beta 0, no
schedule, no beta prior. Reason: jinsu vic_TC_b4 (s3 -> human gains at once, 80M; swing knee about 1.3 N m/rad) reached
knee r 0.95 with residual torque about equal to PD torque at the knee, while every beta ramp from fixed PD here collapsed
when the prior left (knee r 0.5-0.7). Stage 2 unfreezes a (the vic4-tc-scratch-from-vic4cr2-r3 way) from this run.
Logging only: env/vic_gait/{knee,ankle}_r_v08|v10|v14 against the Fukuchi table (HMV-Walk knee r reference).
"""
import dataclasses
from pathlib import Path

from examples.experiments.steering import mlp_human_model_v3_vic4_tc_s5_from_s5_noeff_r1 as _noeff
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_s5_from_s5_noeff_r1 import *  # noqa: F401,F403

FREEZE = dict(v4_freeze_a=True, schedule=False, alpha_init=1.0, beta_init=0.0)
GAIT_R_TABLE = Path(__file__).resolve().parents[4] / "share/human_ref/fukuchi2018_knee_ankle_by_speed.pt"


def env_config(robot_cfg, args):
    cfg = _noeff.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(cfg.control_components["steering"], **FREEZE,
                                                             gait_r_table=str(GAIT_R_TABLE))
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _noeff.agent_config(robot_cfg, env_cfg, args)
