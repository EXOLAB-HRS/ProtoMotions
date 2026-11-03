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
"""vic4_TC_s5_from-ffdv33r2e453_impw10-sc025_r1 (stage 2, local, 2026-11-04; user: stiffness open, no hidden fixed PD, make r_imp
strong enough): from vic4c_TC_s5_from-ffdistillv33r1_r2 epoch_453 (rank 1), the stage-2 recipe
vic4_TC_s5_from-ffdv33r2e453_unfreeze_r1 (stiffness outputs open, alpha 1 / beta 0, s5 feed-forward distillation 0.5,
swing knee out of r_imp, band margin 2 kept) with the impedance prior made strong, no new reward or law:

- vic_imp_prior weight 0.1 -> 1.0 and v4_imp_scale 0.5 -> 0.25 (r_imp = exp(-mean_j (excess_j / scale)^2), excess = ln
  distance outside the human range). Reason (gain logs of unfreeze_r1, 1.4 m/s): the opened policy pushed the frontal hip
  (Hip_x) stiffness below the human range (stance k / k_c 0.60 at e92 -> 0.35 at e276 -> 0.26 at last, below range
  0 -> 71 -> 100 %) while the 1.4 m/s knee r fell 0.78 -> 0.59 -> 0.45; r_imp only fell 0.997 -> 0.936, i.e. 0.006 reward
  per step at weight 0.1. Now one joint 0.35 ln outside the range costs 1 - exp(-(0.35/0.25)^2 / 10) = 0.18 per step.
- Margin 2 is kept (not the band1 hard limit at the range ends): jinsu band1 (margin 1) stayed inside the range by
  construction and still collapsed by its last epoch, so the pull has to come from the prior, not from clipping.
"""
from examples.experiments.steering import mlp_human_model_v3_vic4_tc_s5_from_ffdv33r2e453_unfreeze_r1 as _u
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_s5_from_ffdv33r2e453_unfreeze_r1 import *  # noqa: F401,F403
import dataclasses

IMP_PRIOR_W = 1.0
V4_IMP_SCALE = 0.25


def env_config(robot_cfg, args):
    cfg = _u.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(cfg.control_components["steering"], v4_imp_scale=V4_IMP_SCALE)
    cfg.reward_components["vic_imp_prior"].static_params["weight"] = IMP_PRIOR_W
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _u.agent_config(robot_cfg, env_cfg, args)
