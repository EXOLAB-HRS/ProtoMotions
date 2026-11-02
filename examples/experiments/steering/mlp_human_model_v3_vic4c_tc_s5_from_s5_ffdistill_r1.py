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
"""vic4c_TC_s5_from-s5_ffdistill_r1 (low-stiffness stage 1 with feed-forward distillation, local, 2026-11-02; user:
"A는 바로 구현", "r_imp에서 유각 무릎 빼기"): vic4c_TC_s5_from-s5_noeff_r1 (s5 -> human impedance centre at once, a frozen,
no VIC effort) plus two changes:

- Feed-forward distillation (PPO ff_distill): the frozen fixed-PD teacher se-v1-s5 runs on the same observations every
  control step; the env stores the torque the teacher's PD targets would give with the fixed PD gains in this state
  minus this step's PD torque, per VIC joint, and the actor's 10 feed-forward outputs are pulled toward it (in units of
  the feed-forward cap, weight FFD_WEIGHT * MSE). Reason: in the noeff_r1 control the feed-forward head grew to 1/25 of
  the PD-target head in 15M (sigma 0.055, lr 2e-5) while the knee had almost no PD torque (RMS 0.06 vs WBDS 0.20 N m/kg).
  The target comes from the teacher policy, not from a walking table, so it covers standing, starts and turns. The
  30 Hz feed-forward cannot follow the target's fast part; that part stays with stiffness. Hidden stiffness (tau_ff vs
  joint error) is checked in evaluation.
- r_imp without the swing knee (v4_imp_skip_swing_knee): the plant's passive knee stiffness on human swing angles
  (6-24 N m/rad) is above the human swing reference (0.5-3.5), a penalty no action could remove.

Teacher checkpoint: env HC_FF_DISTILL_TEACHER (se-v1-s5 last.ckpt, SHA b6075429...).
"""
import dataclasses
import os

from examples.experiments.steering import mlp_human_model_v3_vic4c_tc_s5_from_s5_noeff_r1 as _s1
from examples.experiments.steering.mlp_human_model_v3_vic4c_tc_s5_from_s5_noeff_r1 import *  # noqa: F401,F403

FFD_WEIGHT = 0.5


def env_config(robot_cfg, args):
    cfg = _s1.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(cfg.control_components["steering"], v4_imp_skip_swing_knee=True)
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    cfg = _s1.agent_config(robot_cfg, env_cfg, args)
    teacher = os.environ.get("HC_FF_DISTILL_TEACHER", "")
    if not teacher or not os.path.isfile(teacher):
        raise FileNotFoundError(f"HC_FF_DISTILL_TEACHER must name the fixed-PD teacher checkpoint, got {teacher!r}")
    cfg.ff_distill.weight = FFD_WEIGHT
    cfg.ff_distill.teacher_checkpoint = teacher
    return cfg
