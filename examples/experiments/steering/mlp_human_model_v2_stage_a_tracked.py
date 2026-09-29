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
"""Stage A again, against the two defects the first Stage A run measured.

The command range, plant, gains, termination and 0.5/0.5 reward split are
inherited unchanged from the Stage A experiment beside this file; widening the
command range waits until Stage A is actually met.

Replaying that run's policy showed it travels in the commanded direction
(cos 0.97, no falls) but misses the commanded speed by 0.32 m/s on average and
0.81 m/s at p95, while the discriminator reached 0.9996 accuracy and the style
reward decayed from 0.68 to 0.09. Two settings therefore differ:

* The steering reward's ``vel_err_scale`` goes from the kernel default 0.25 to
  1.5, the value the reference speed-tracking experiment uses. At 0.25 a 0.4 m/s
  undershoot costs about 4% of the reward, so speed error carried almost no
  gradient and the policy settled on a gait rather than on the command.
* The discriminator learns at half the rate. It is the side that won, and its
  reward was the signal that vanished.

The expert library is the same 24 clips with weights proportional to their
frames inside the commanded 0.8-1.2 m/s band. Motion weights drive both the
expert batches and reference-state initialisation, and under the previous
uniform-per-class weights only 38.8% of expert frames sat in the band while a
2.4 m/s clip took a sixth of all expert samples.
"""

import importlib.util
from pathlib import Path

_STAGE_A_PATH = Path(__file__).with_name("mlp_human_model_v2_stage_a_forward.py")
_SPEC = importlib.util.spec_from_file_location("human_model_v2_stage_a", _STAGE_A_PATH)
_STAGE_A = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_STAGE_A)

terrain_config = _STAGE_A.terrain_config
scene_lib_config = _STAGE_A.scene_lib_config
motion_lib_config = _STAGE_A.motion_lib_config
apply_inference_overrides = _STAGE_A.apply_inference_overrides
configure_robot_and_simulator = _STAGE_A.configure_robot_and_simulator

VELOCITY_ERROR_SCALE = 1.5
DISCRIMINATOR_LEARNING_RATE = 5e-5


def env_config(robot_cfg, args):
    cfg = _STAGE_A.env_config(robot_cfg, args)
    cfg.reward_components["heading_rew"].static_params["vel_err_scale"] = VELOCITY_ERROR_SCALE
    return cfg


def agent_config(robot_config, env_config, args):
    cfg = _STAGE_A.agent_config(robot_config, env_config, args)
    cfg.model.discriminator_optimizer.lr = DISCRIMINATOR_LEARNING_RATE
    return cfg
