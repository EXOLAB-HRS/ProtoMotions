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
"""vic4c_TC_s5_from-ffdistillr1_r2 (local, 2026-11-03; server instruction: keep the v3.1 distillation run as the baseline
and continue it with the same settings and seed past 11 KST): continuation of vic4c_TC_s5_from-s5_ffdistill_r1 from its
last.ckpt. The recipe is that run's recipe unchanged (plant v3.1, stiffness outputs frozen at the human centre, alpha 1 /
beta 0, no VIC effort, feed-forward distillation from the frozen se-v1-s5, r_imp without the swing knee); only the parent
differs (79 outputs, so no rows are appended).
"""
from examples.experiments.steering import mlp_human_model_v3_vic4c_tc_s5_from_s5_ffdistill_r1 as _r1
from examples.experiments.steering.mlp_human_model_v3_vic4c_tc_s5_from_s5_ffdistill_r1 import *  # noqa: F401,F403


def env_config(robot_cfg, args):
    return _r1.env_config(robot_cfg, args)


def agent_config(robot_cfg, env_cfg, args):
    return _r1.agent_config(robot_cfg, env_cfg, args)
