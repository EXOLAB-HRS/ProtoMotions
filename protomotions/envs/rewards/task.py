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
"""Task-specific reward compute kernels.

Pure tensor functions (kernels) for computing task-specific rewards.
Use MdpComponent in experiment configs to bind kernels to context paths:

    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.rewards.task import compute_heading_velocity_rew
    
    reward_components = {
        "heading_velocity": MdpComponent(
            compute_func=compute_heading_velocity_rew,
            dynamic_vars={
                "root_pos": EnvContext.current.root_pos,
                "prev_root_pos": EnvContext.steering.prev_root_pos,
                "root_rot": EnvContext.current.root_rot,
                "tar_dir": EnvContext.steering.tar_dir,
                "tar_speed": EnvContext.steering.tar_speed,
                "tar_face_dir": EnvContext.steering.tar_face_dir,
                "dt": EnvContext.dt,
            },
        ),
    }

Provides reward functions for specific tasks:
- Steering/locomotion rewards
- Path following rewards
"""

import torch
from torch import Tensor

from protomotions.utils.rotations import calc_heading_quat, quat_rotate


# =============================================================================
# Steering Reward Kernels
# =============================================================================

def compute_heading_velocity_rew(
    root_pos: Tensor,
    prev_root_pos: Tensor,
    root_rot: Tensor,
    tar_dir: Tensor,
    tar_speed: Tensor,
    tar_face_dir: Tensor,
    dt: float,
    vel_err_scale: float = 0.25,
) -> Tensor:
    """Reward for moving in target direction at target speed while facing that direction.

    Computes weighted combination of:
    - Direction reward: exponential penalty on velocity error and tangent velocity
    - Facing reward: alignment between robot heading and target direction

    Args:
        root_pos: Current root position [num_envs, 3].
        prev_root_pos: Previous root position [num_envs, 3].
        root_rot: Root orientation quaternions [num_envs, 4] (w-last).
        tar_dir: Target movement direction [num_envs, 2].
        tar_speed: Target speed [num_envs].
        tar_face_dir: Target facing direction [num_envs, 2] (can differ from tar_dir).
        dt: Simulation timestep.

    Returns:
        Reward [num_envs] in range [0, 1].
    """

    # vel_err_scale controls how sharply speed error is penalized; the default (0.25) is soft
    # (a 0.4 m/s undershoot costs only ~4% reward), which lets the policy quantize to the
    # reference gaits instead of tracking commanded speed. Raise it (via the reward component's
    # static_params) to make speed tracking a real gradient. See mlp_speed.py.
    tangent_err_w = 0.1

    dir_reward_w = 0.7
    facing_reward_w = 0.3

    # Compute velocity in target direction
    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)

    # Compute tangent (perpendicular) velocity
    tar_dir_vel = tar_dir_speed.unsqueeze(-1) * tar_dir
    tangent_vel = root_vel[..., :2] - tar_dir_vel
    tangent_vel_err = torch.sum(torch.square(tangent_vel), dim=-1)

    # Direction reward: penalize velocity error and tangent movement
    tar_vel_err = tar_speed - tar_dir_speed
    dir_reward = torch.exp(
        -vel_err_scale * (tar_vel_err * tar_vel_err + tangent_err_w * tangent_vel_err)
    )

    # Zero reward for moving backwards
    speed_mask = tar_dir_speed <= 0
    dir_reward[speed_mask] = 0

    # Facing reward: robot should face the target facing direction
    heading_rot = calc_heading_quat(root_rot, w_last=True)
    facing_dir = torch.zeros_like(root_pos)
    facing_dir[..., 0] = 1.0
    facing_dir = quat_rotate(heading_rot, facing_dir, w_last=True)

    facing_err = torch.sum(tar_face_dir * facing_dir[..., 0:2], dim=-1)
    facing_reward = torch.clamp_min(facing_err, 0.0)

    reward = dir_reward_w * dir_reward + facing_reward_w * facing_reward

    return reward


def compute_split_heading_velocity_rew(
    root_pos: Tensor,
    prev_root_pos: Tensor,
    root_rot: Tensor,
    tar_dir: Tensor,
    tar_speed: Tensor,
    tar_face_dir: Tensor,
    dt: float,
    speed_err_scale: float = 0.5,
    tangent_err_scale: float = 0.025,
    speed_reward_w: float = 0.35,
    direction_reward_w: float = 0.35,
    facing_reward_w: float = 0.30,
    upright_reward_w: float = 0.0,
    upright_height_min: float = 0.5,
    upright_height_margin: float = 0.4,
) -> Tensor:
    """Steering reward with independently tunable speed and direction channels.

    Unlike :func:`compute_heading_velocity_rew`, sharpening speed tracking here does
    not also sharpen the tangent-velocity penalty.  This is intended for curriculum
    fine-tuning of an already stable steering policy.
    """
    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    projected_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)

    projected_velocity = projected_speed.unsqueeze(-1) * tar_dir
    tangent_velocity = root_vel[..., :2] - projected_velocity
    tangent_error = torch.sum(torch.square(tangent_velocity), dim=-1)

    speed_error = tar_speed - projected_speed
    speed_reward = torch.exp(-speed_err_scale * torch.square(speed_error))
    direction_reward = torch.exp(-tangent_err_scale * tangent_error)

    # Moving opposite the command must not earn speed/direction credit.
    forward_mask = projected_speed > 0
    speed_reward = speed_reward * forward_mask
    direction_reward = direction_reward * forward_mask

    heading_rot = calc_heading_quat(root_rot, w_last=True)
    facing_dir = torch.zeros_like(root_pos)
    facing_dir[..., 0] = 1.0
    facing_dir = quat_rotate(heading_rot, facing_dir, w_last=True)
    facing_alignment = torch.sum(tar_face_dir * facing_dir[..., 0:2], dim=-1)
    facing_reward = torch.clamp_min(facing_alignment, 0.0)

    motion_reward = (
        speed_reward_w * speed_reward
        + direction_reward_w * direction_reward
        + facing_reward_w * facing_reward
    )

    # The evaluation stability gate defines upright as root height > 0.5 m.
    # A short linear margin supplies a useful gradient above that boundary while
    # keeping upright_reward_w=0 exactly backward-compatible with older configs.
    upright_reward = torch.clamp(
        (root_pos[..., 2] - upright_height_min) / upright_height_margin,
        min=0.0,
        max=1.0,
    )
    return (1.0 - upright_reward_w) * motion_reward + upright_reward_w * upright_reward


def compute_foot_plant_rew(
    rigid_body_pos: Tensor,
    rigid_body_vel: Tensor,
    foot_body_ids: Tensor,
    contact_height: float = 0.12,
    contact_margin: float = 0.04,
    velocity_scale: float = 4.0,
    aggregation: str = "mean",
) -> Tensor:
    """Reward stationary feet only while they are near the flat ground.

    A soft height mask avoids requiring expensive contact sensors during
    training.  Swinging feet receive no penalty: when no selected foot body is
    near the ground, the component returns the neutral reward one.  This kernel
    is intended for the locomotion experiments, whose terrain is explicitly
    flat at z=0.
    """
    foot_pos = torch.index_select(rigid_body_pos, 1, foot_body_ids)
    foot_vel = torch.index_select(rigid_body_vel, 1, foot_body_ids)

    # 1 below contact_height, linearly fading to 0 over contact_margin.
    near_ground = torch.clamp(
        (contact_height + contact_margin - foot_pos[..., 2]) / contact_margin,
        min=0.0,
        max=1.0,
    )
    horizontal_speed_sq = torch.sum(torch.square(foot_vel[..., :2]), dim=-1)
    stationary = torch.exp(-velocity_scale * horizontal_speed_sq)

    mask_sum = torch.sum(near_ground, dim=-1)
    if aggregation == "mean":
        planted_reward = torch.sum(near_ground * stationary, dim=-1) / torch.clamp_min(
            mask_sum, 1.0e-6
        )
    elif aggregation == "strict_min":
        # A mean lets one stationary ankle/toe hide another contact body's
        # skating.  Softly blend non-contact bodies toward neutral one, then
        # score the worst likely-contact body.  The height mask remains smooth.
        effective_stationary = 1.0 - near_ground * (1.0 - stationary)
        planted_reward = torch.min(effective_stationary, dim=-1).values
    else:
        raise ValueError(f"unsupported foot-plant aggregation: {aggregation!r}")
    return torch.where(mask_sum > 0.0, planted_reward, torch.ones_like(planted_reward))


# =============================================================================
# Path Following Reward Kernels
# =============================================================================

def compute_path_following_rew(
    head_pos: Tensor,
    tar_pos: Tensor,
    height_conditioned: bool,
    pos_err_scale: float = 2.0,
    height_err_scale: float = 10.0,
) -> Tensor:
    """Reward for following a path (staying close to target position).

    Computes exponential reward based on:
    - Horizontal distance to target position
    - Optionally: vertical distance to target position

    Args:
        head_pos: Current head position [num_envs, 3] (ground-relative).
        tar_pos: Target position from path [num_envs, 3] (ground-relative).
        height_conditioned: Whether to include height in reward.
        pos_err_scale: Coefficient for position error.
        height_err_scale: Coefficient for height error.

    Returns:
        Reward [num_envs] in range [0, 1].
    """
    pos_diff = tar_pos[..., 0:2] - head_pos[..., 0:2]
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    height_diff = tar_pos[..., 2] - head_pos[..., 2]
    height_err = height_diff * height_diff

    pos_reward = torch.exp(-pos_err_scale * pos_err)
    height_reward = torch.exp(-height_err_scale * height_err)

    if height_conditioned:
        # Multiplicative reward ensures both terms are properly met.
        reward = pos_reward * height_reward
    else:
        reward = pos_reward

    return reward


__all__ = [
    "compute_heading_velocity_rew",
    "compute_split_heading_velocity_rew",
    "compute_path_following_rew",
]
