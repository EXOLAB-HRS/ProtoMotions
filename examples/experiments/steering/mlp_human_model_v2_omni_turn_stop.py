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
"""Steering teacher trained from scratch on the human_model_v2 plant.

Task scope is omnidirectional travel, turning and stopping. The reward,
network and optimiser follow the settings the 69-coordinate steering policy
converged with; only what the 59-coordinate plant forces to differ is changed:

* ``--robot-name human_model_v2`` selects the v2 assets, ROM and strength
  profile. The 69-coordinate policies and checkpoints are not compatible.
* Control switches from the v2 default ``TORQUE`` to ``PROPORTIONAL``. The
  plant intercepts control in ``HumanJointModel.apply``: under PROPORTIONAL it
  forms the PD request itself and then applies the direction-, angle- and
  velocity-dependent strength caps plus the passive torques. The policy
  therefore emits ROM-normalised joint targets, exactly like the v1 teacher,
  while the physiological ceiling still binds.
* Stop commands need a reward kernel that scores standing still; the split
  kernel gates its speed channel on forward progress and pays nothing for a
  correct stop.
* Commanded speed is capped at 1.2 m/s. v2 carries measured human torque
  limits instead of the 500 Nm placeholder, and six hip/knee directions
  already fail to reach the passive ROM endpoint under those limits.
* A root-height termination, an upright reward channel and a raised
  motion-start probability were added after the first run stalled: with no
  termination a collapsed character ran out its whole episode on facing
  credit alone.

Expert motions must be the retargeted library produced by
``human_model_v2.adapters.smpl_retarget``; a 24-body SMPL library has the wrong
body set for the discriminator.
"""

import importlib.util
import re
from pathlib import Path

import torch

from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.motion_manager.config import MotionManagerConfig
from protomotions.envs.rewards.task import (
    compute_foot_plant_rew,
    compute_split_heading_velocity_stop_rew,
)
from protomotions.envs.terminations import height_termination
from protomotions.robot_configs.base import ControlType

_BASE_PATH = Path(__file__).with_name("mlp.py")
_SPEC = importlib.util.spec_from_file_location("steering_mlp_base", _BASE_PATH)
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

scene_lib_config = _BASE.scene_lib_config
motion_lib_config = _BASE.motion_lib_config
apply_inference_overrides = _BASE.apply_inference_overrides

# Fallback gains for the ten upper-body coordinates that carry no active
# strength; a strength-scaled gain is undefined for them.
PD_GAINS = (
    (r".*_(Hip|Knee|Ankle|Subtalar)_.*", 800.0, 80.0),
    (r".*_Toe_.*", 500.0, 50.0),
    (r"(Torso|Spine|Chest)_.*", 1000.0, 100.0),
    (r"(Neck|Head|.*_Thorax|.*_Shoulder|.*_Elbow)_.*", 500.0, 50.0),
    (r".*_(Wrist|Hand)_.*", 300.0, 30.0),
)

# Every coordinate with a strength budget gets stiffness = cap / SATURATION_ANGLE,
# so SATURATION_ANGLE is the tracking error at which that coordinate maxes out.
# Swept on the plant against the retargeted walking reference (jobs 902/903/904):
# v1's carried-over stiffness 800 sits at a 0.27 rad saturation angle and clips
# 89-93% of substeps; 1.5 rad gives the lowest leg tracking RMSE (0.233 rad) and
# cuts clipping to 0.37. Adding gravity compensation moved these by under 0.01,
# so the ceiling contact is not a missing feedforward term.
SATURATION_ANGLE_RAD = 1.5
DAMPING_TIME_S = 0.1

TARGET_SPEED_MIN = 0.5
TARGET_SPEED_MAX = 1.2
STOP_PROBABILITY = 0.2

# A first run without these three settings stalled: episodes never terminated,
# so a collapsed character kept collecting the facing channel for 300 steps
# while the style reward decayed. The height matches the project's existing
# upright definition (root above 0.5 m).
TERMINATION_ROOT_HEIGHT = 0.5
UPRIGHT_REWARD_W = 0.1
MOTION_START_PROBABILITY = 0.5


def configure_robot_and_simulator(robot_cfg, simulator_cfg, args):
    profile = getattr(robot_cfg, "human_model_profile", None)
    if profile != "human_model_v2":
        raise ValueError(
            f"this experiment requires --robot-name human_model_v2, got profile {profile!r}"
        )

    robot_cfg.control.control_type = ControlType.PROPORTIONAL
    for name, info in robot_cfg.control.control_info.items():
        cap = float(info.effort_limit or 0.0)
        if cap > 0.0:
            info.stiffness = cap / SATURATION_ANGLE_RAD
            info.damping = info.stiffness * DAMPING_TIME_S
            continue
        for pattern, stiffness, damping in PD_GAINS:
            if re.fullmatch(pattern, name):
                info.stiffness = stiffness
                info.damping = damping
                break
        else:
            raise ValueError(f"no PD gain group matches coordinate {name!r}")


def terrain_config(args):
    cfg = _BASE.terrain_config(args)
    cfg.spacing_between_scenes = 1.0
    return cfg


def env_config(robot_cfg, args):
    cfg = _BASE.env_config(robot_cfg, args)
    body_names = robot_cfg.kinematic_info.body_names

    steering = cfg.control_components["steering"]
    steering.tar_speed_min = TARGET_SPEED_MIN
    steering.tar_speed_max = TARGET_SPEED_MAX
    steering.heading_change_steps_min = 150
    steering.heading_change_steps_max = 301
    steering.random_heading_probability = 0.2
    steering.random_speed_probability = 1.0
    steering.standard_heading_change = 0.5
    steering.standard_speed_change = 0.35
    steering.stop_probability = STOP_PROBABILITY
    # Independent travel and facing directions are what separate genuine
    # lateral and backward walking from a rotated forward walk.
    steering.enable_rand_facing = True

    cfg.reward_components["heading_rew"] = MdpComponent(
        compute_func=compute_split_heading_velocity_stop_rew,
        dynamic_vars={
            "root_pos": EnvContext.current.root_pos,
            "prev_root_pos": EnvContext.steering.prev_root_pos,
            "root_rot": EnvContext.current.root_rot,
            "tar_dir": EnvContext.steering.tar_dir,
            "tar_speed": EnvContext.steering.tar_speed,
            "tar_face_dir": EnvContext.steering.tar_face_dir,
            "dt": EnvContext.dt,
        },
        static_params={
            "weight": 1.0,
            "speed_err_scale": 8.0,
            "tangent_err_scale": 0.025,
            "speed_reward_w": 0.35,
            "direction_reward_w": 0.35,
            "facing_reward_w": 0.30,
            "upright_reward_w": UPRIGHT_REWARD_W,
        },
    )

    # Excluding every body but the pelvis turns the generic body-height check
    # into a root-height check. "threshold" cannot be used here: MdpComponent
    # reserves that key for reward/termination metadata and never forwards it.
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
    cfg.motion_manager = MotionManagerConfig(init_start_prob=MOTION_START_PROBABILITY)

    foot_names = (
        robot_cfg.common_naming_to_robot_body_names["all_left_foot_bodies"]
        + robot_cfg.common_naming_to_robot_body_names["all_right_foot_bodies"]
    )
    cfg.reward_components["foot_plant_rew"] = MdpComponent(
        compute_func=compute_foot_plant_rew,
        dynamic_vars={
            "rigid_body_pos": EnvContext.current.rigid_body_pos,
            "rigid_body_vel": EnvContext.current.rigid_body_vel,
        },
        static_params={
            "foot_body_ids": torch.tensor(
                [body_names.index(name) for name in foot_names], dtype=torch.long
            ),
            "contact_height": 0.12,
            "contact_margin": 0.04,
            "velocity_scale": 4.0,
            "weight": 0.05,
        },
    )
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    cfg = _BASE.agent_config(robot_cfg, env_cfg, args)
    cfg.task_reward_w = 0.7
    return cfg
