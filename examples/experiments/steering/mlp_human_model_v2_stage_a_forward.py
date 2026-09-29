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
#
"""Stage A of the AMP + steering guide on the human_model_v2 plant.

The guide's staged start for a new human model is forward walking at 0.8-1.2
m/s with the facing direction locked to travel, no heading change and no stop;
speed, heading, stop and independent facing widen only in later stages. An
earlier attempt trained the full command range from scratch and never learned
to travel, which is the failure the staging exists to avoid.

Everything the guide specifies is inherited unchanged from the reference
steering experiment: the actor sees state, command and an 8-frame history, the
discriminator sees history alone, reward splits 0.5 task and 0.5 style, and the
network, optimiser and episode length are untouched. Only the command range is
narrowed to Stage A.

Two settings depart from the reference experiment, both because the plant is a
measured human rather than the 500 Nm placeholder:

* PD gains come from a sweep on this plant, not from the reference file. The
  guide's own failure checklist puts actuator strength and PD gain first when
  the character cannot walk, and the sweep found the inherited stiffness clipped
  the strength ceiling on 89-93% of substeps.
* A root-height termination scores a fall. Without it the reference setup leaves
  a collapsed character running its full episode and collecting facing reward,
  which is what the first two runs did.

The expert motions are the retargeted locomotion clips minus the backward and
lateral ones, since Stage A never commands those directions and the
discriminator would otherwise reward a style the task cannot produce.
"""

import importlib.util
from pathlib import Path

import torch

from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.terminations import height_termination

_BASE_PATH = Path(__file__).with_name("mlp.py")
_SPEC = importlib.util.spec_from_file_location("steering_mlp_base", _BASE_PATH)
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

_PLANT_PATH = Path(__file__).with_name("mlp_human_model_v2_omni_turn_stop.py")
_PLANT_SPEC = importlib.util.spec_from_file_location("human_model_v2_plant", _PLANT_PATH)
_PLANT = importlib.util.module_from_spec(_PLANT_SPEC)
_PLANT_SPEC.loader.exec_module(_PLANT)

terrain_config = _BASE.terrain_config
scene_lib_config = _BASE.scene_lib_config
motion_lib_config = _BASE.motion_lib_config
agent_config = _BASE.agent_config
apply_inference_overrides = _BASE.apply_inference_overrides
configure_robot_and_simulator = _PLANT.configure_robot_and_simulator

TARGET_SPEED_MIN = 0.8
TARGET_SPEED_MAX = 1.2
TERMINATION_ROOT_HEIGHT = 0.5


def env_config(robot_cfg, args):
    cfg = _BASE.env_config(robot_cfg, args)

    steering = cfg.control_components["steering"]
    steering.tar_speed_min = TARGET_SPEED_MIN
    steering.tar_speed_max = TARGET_SPEED_MAX
    # Zero heading change with zero random-heading probability holds the command
    # on the initial direction, which is what Stage A means by straight only.
    steering.random_heading_probability = 0.0
    steering.standard_heading_change = 0.0
    steering.random_speed_probability = 1.0
    steering.stop_probability = 0.0
    steering.enable_rand_facing = False

    body_names = robot_cfg.kinematic_info.body_names
    pelvis_index = body_names.index("Pelvis")
    cfg.termination_components["fallen"] = MdpComponent(
        compute_func=height_termination,
        dynamic_vars={
            "rigid_body_pos": EnvContext.current.rigid_body_pos,
            "ground_heights": EnvContext.ground_heights,
        },
        static_params={
            "termination_height": TERMINATION_ROOT_HEIGHT,
            "non_termination_body_ids": torch.tensor(
                [i for i in range(len(body_names)) if i != pelvis_index], dtype=torch.long
            ),
        },
    )
    return cfg
