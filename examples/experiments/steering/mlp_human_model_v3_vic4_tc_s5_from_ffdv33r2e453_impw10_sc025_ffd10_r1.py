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
"""vic4_TC_s5_from-ffdv33r2e453_impw10-sc025-ffd10_r1 (stage 2, local, 2026-11-04): ..._impw10_sc025_r1 with the feed-forward
distillation weight 0.5 -> 1.0. Reason: with the stiffness held in the human range and no hidden fixed PD, the fast knee
torque the fixed-PD teacher gets from its high gains must come from tau_ff (server diagnosis 11-03: "강성을 학습하게 두면
무릎 빠른 토크가 모자란다"); the distillation fit stayed low (R^2 0.10-0.14 in band1, mse 0.28-0.33 in unfreeze_r1).
"""
from examples.experiments.steering import mlp_human_model_v3_vic4_tc_s5_from_ffdv33r2e453_impw10_sc025_r1 as _i
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_s5_from_ffdv33r2e453_impw10_sc025_r1 import *  # noqa: F401,F403

FFD_WEIGHT_L2 = 1.0


def env_config(robot_cfg, args):
    return _i.env_config(robot_cfg, args)


def agent_config(robot_cfg, env_cfg, args):
    cfg = _i.agent_config(robot_cfg, env_cfg, args)
    cfg.ff_distill.weight = FFD_WEIGHT_L2
    return cfg
