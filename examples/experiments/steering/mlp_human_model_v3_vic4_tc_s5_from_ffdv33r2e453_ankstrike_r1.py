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
"""WITHDRAWN 2026-11-03 16:40 (server check: the -1 deg target is a Fukuchi absolute angle; with the standing pose
removed the ankle already matches people). Kept for the stopped run; use ..._from_ffdv33r2e453_unfreeze_r1.

vic4_TC_s5_from-ffdv33r2e453_ankstrike_r1 (stage 2, local, 2026-11-03; user: "전부 다 적용해서 진행하자"): from
vic4c_TC_s5_from-ffdistillv33r1_r2 epoch_453 (plant v3.3, first W2 v5 + W3 pass of the series), the server's stage-2 law
vic4_TC_s5_from-s5_ffdistill_r1 (stiffness outputs unfrozen, alpha 1 / beta 0, feed-forward distillation from the frozen
se-v1-s5, swing knee out of r_imp) plus one reward term:

- ankle_strike_prior (knee_strike_prior on the ankle dofs, same heel-strike window gait % > 85 or < 5): ankle angle near
  ANKLE_TARGET_DEG (raw dof sign; -1 = 1 deg dorsiflexion, Fukuchi 2018 heel strike +0.4 / +0.9 / +1.8 deg at
  0.8 / 1.0 / 1.4 m/s), sigma ANKLE_SIGMA_DEG, weight ANKLE_STRIKE_W. Reason: s5 and every run built on it land
  10-12 deg plantarflexed (cycle-mean bias -10 to -13 deg) and plant v3.3, distillation and 179M of training did not
  move it; human_gait_prior's 3 x SD kernel gives little gradient there.

The parent has 79 outputs already, so no rows are appended.
"""
from examples.experiments.steering import mlp_human_model_v3_vic4_tc_s5_from_s5_ffdistill_r1 as _s2
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_s5_from_s5_ffdistill_r1 import *  # noqa: F401,F403
from protomotions.envs.context_views import EnvContext
from protomotions.envs.control.cadence_gait_prior import GAIT_PCT, GAIT_PCT_VALID, knee_strike_prior
from protomotions.envs.mdp_component import MdpComponent

ANKLE_TARGET_DEG = -1.0
ANKLE_SIGMA_DEG = 6.0
ANKLE_STRIKE_W = 0.3


def env_config(robot_cfg, args):
    cfg = _s2.env_config(robot_cfg, args)
    dofs = list(robot_cfg.kinematic_info.dof_names)
    knee = cfg.reward_components["knee_strike_prior"].static_params
    cfg.reward_components["ankle_strike_prior"] = MdpComponent(
        compute_func=knee_strike_prior,
        dynamic_vars={"dof_pos": EnvContext.current.dof_pos, "gait_pct": GAIT_PCT, "gait_pct_valid": GAIT_PCT_VALID},
        static_params={"knee_dofs": [dofs.index("L_Ankle_y"), dofs.index("R_Ankle_y")], "ratio_tol": 0.0,
                       "late_pct": knee["late_pct"], "early_pct": knee["early_pct"],
                       "target_deg": ANKLE_TARGET_DEG, "sigma_deg": ANKLE_SIGMA_DEG, "weight": ANKLE_STRIKE_W})
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _s2.agent_config(robot_cfg, env_cfg, args)
