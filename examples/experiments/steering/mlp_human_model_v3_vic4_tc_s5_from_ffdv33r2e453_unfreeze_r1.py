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
"""vic4_TC_s5_from-ffdv33r2e453_unfreeze_r1 (stage 2, local, 2026-11-03; server instruction 16:40): from
vic4c_TC_s5_from-ffdistillv33r1_r2 epoch_453 (plant v3.3, W2 v5 + W3 pass), the server's stage-2 law
vic4_TC_s5_from-s5_ffdistill_r1 unchanged (stiffness outputs unfrozen, alpha 1 / beta 0, feed-forward distillation from
the frozen se-v1-s5, swing knee out of r_imp). No ankle term: the ankle_strike_prior run (from-ffdv33r2e453_ankstrike_r1)
was withdrawn because its target came from Fukuchi absolute angles (standing ankle +9.4 deg), while with the standing
pose removed the ankle offset already matches people (W2 v5 ankle_dorsiflexion_offset). The parent has 79 outputs.
"""
from examples.experiments.steering import mlp_human_model_v3_vic4_tc_s5_from_s5_ffdistill_r1 as _s2
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_s5_from_s5_ffdistill_r1 import *  # noqa: F401,F403


def env_config(robot_cfg, args):
    return _s2.env_config(robot_cfg, args)


def agent_config(robot_cfg, env_cfg, args):
    return _s2.agent_config(robot_cfg, env_cfg, args)
