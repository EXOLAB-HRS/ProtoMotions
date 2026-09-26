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
"""p2f3a: a12 head recipe + knee gait shape term (naturalness), first steering teacher.

Starts from p2f1. Command and stop settings are launcher overrides. Knee
bounds come from the v4 train pack forward clips at 0.8-1.25 m/s, split by
ankle horizontal speed: stance knee p75 7.6 deg (IQR 3.0-7.6), swing knee
median 43.2 deg (IQR 26.3-53.6); scales are the IQR and half the swing IQR.
Weights: heading 0.75, a12 head 0.15, knee 0.10. Settings table:
share/training_configs/p2_series_server/README.md.
"""

import importlib.util
import math
from pathlib import Path

import torch

from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.rewards.locomotion_quality import knee_gait_shape


_SPEC = importlib.util.spec_from_file_location(
    "knee_gait_a12_base", Path(__file__).with_name("mlp_human_model_v3_a12_scratch_head.py")
)
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

EXPERIMENT_ID = "p2f3a"
HEADING_WEIGHT = 0.75
KNEE_WEIGHT = 0.10
STANCE_KNEE_MAX_DEG, STANCE_SCALE_DEG = 7.6, 4.6
SWING_KNEE_MIN_DEG, SWING_SCALE_DEG = 43.2, 13.65

terrain_config = _BASE.terrain_config
scene_lib_config = _BASE.scene_lib_config
motion_lib_config = _BASE.motion_lib_config
agent_config = _BASE.agent_config
apply_inference_overrides = _BASE.apply_inference_overrides
configure_robot_and_simulator = _BASE.configure_robot_and_simulator


def env_config(robot_cfg, args):
    cfg = _BASE.env_config(robot_cfg, args)
    names = robot_cfg.kinematic_info.body_names
    dofs = robot_cfg.kinematic_info.dof_names
    cfg.reward_components["heading_rew"].static_params["weight"] = HEADING_WEIGHT
    cfg.reward_components["knee_gait_shape"] = MdpComponent(
        compute_func=knee_gait_shape,
        dynamic_vars={"dof_pos": EnvContext.current.dof_pos,
                      "rigid_body_vel": EnvContext.current.rigid_body_vel},
        static_params={
            "knee_dof_ids": torch.tensor([dofs.index("L_Knee_y"), dofs.index("R_Knee_y")], dtype=torch.long),
            "ankle_body_ids": torch.tensor([names.index("L_Ankle"), names.index("R_Ankle")], dtype=torch.long),
            "stance_knee_max_rad": math.radians(STANCE_KNEE_MAX_DEG),
            "stance_scale_rad": math.radians(STANCE_SCALE_DEG),
            "swing_knee_min_rad": math.radians(SWING_KNEE_MIN_DEG),
            "swing_scale_rad": math.radians(SWING_SCALE_DEG),
            "weight": KNEE_WEIGHT,
        },
    )
    return cfg
