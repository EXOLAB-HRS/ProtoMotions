"""Discriminator observations on a subset of bodies (teacher_se_v1 AMP fix candidate, 2026-10-26).

Same layout as compute_historical_max_coords_from_state / _from_motion_lib (root height, local body positions without
the root, tan-norm rotations, velocities, angular velocities, per history step), computed only for the bodies in
body_ids. body_ids[0] must be the root (0). Why: on se-v1-s3 the passive joints (Elbow_x/y, Hand_x/y/z: kp = 0,
effort 0) wobble with joint-velocity SDs 11-82 x the motion pack's, and the wrist/forearm/hand bodies give the AMP
discriminator an easy cue (agent_acc 0.98, pos_acc 1.0 in every run). Restricting the discriminator to the bodies
the gait judges look at removes that cue without changing the plant.
"""
from typing import List, Union

import torch
from torch import Tensor

from protomotions.envs.obs.humanoid import compute_humanoid_max_coords_observations
from protomotions.envs.obs.humanoid_historical import compute_historical_max_coords_from_state


def compute_historical_max_coords_subset_from_state(
    historical_rigid_body_pos: Tensor, historical_rigid_body_rot: Tensor, historical_rigid_body_vel: Tensor,
    historical_rigid_body_ang_vel: Tensor, historical_ground_heights: Tensor, historical_body_contacts: Tensor,
    body_ids: List[int], local_obs: bool = True, root_height_obs: bool = True, w_last: bool = True,
    history_steps: Union[int, List[int]] = None,
) -> Tensor:
    if historical_rigid_body_pos is None:
        raise ValueError("Historical state tensors are None (num_state_history_steps must be > 0).")
    idx = torch.as_tensor(body_ids, device=historical_rigid_body_pos.device)
    return compute_historical_max_coords_from_state(
        historical_rigid_body_pos[:, :, idx], historical_rigid_body_rot[:, :, idx],
        historical_rigid_body_vel[:, :, idx], historical_rigid_body_ang_vel[:, :, idx],
        historical_ground_heights, historical_body_contacts, local_obs=local_obs, root_height_obs=root_height_obs,
        observe_contacts=False, w_last=w_last, history_steps=history_steps)


def compute_historical_max_coords_subset_from_motion_lib(
    motion_lib, motion_ids: Tensor, motion_times: Tensor, num_state_history_steps: int, dt: float,
    body_ids: List[int], local_obs: bool = True, root_height_obs: bool = True,
    history_steps: Union[int, List[int]] = None,
) -> Tensor:
    n = motion_ids.shape[0]
    device = motion_ids.device
    if history_steps is None:
        steps = list(range(1, num_state_history_steps + 1))
    elif isinstance(history_steps, int):
        steps = list(range(1, history_steps + 1))
    else:
        steps = list(history_steps)
    idx = torch.as_tensor(body_ids, device=device)
    obs = []
    for k in steps:
        s = motion_lib.get_motion_state(motion_ids, (motion_times - k * dt).clamp(min=0.0))
        obs.append(compute_humanoid_max_coords_observations(
            body_pos=s.rigid_body_pos[:, idx], body_rot=s.rigid_body_rot[:, idx], body_vel=s.rigid_body_vel[:, idx],
            body_ang_vel=s.rigid_body_ang_vel[:, idx], ground_height=torch.zeros(n, 1, device=device),
            body_contacts=torch.zeros(n, len(body_ids), dtype=torch.bool, device=device), local_obs=local_obs,
            root_height_obs=root_height_obs, observe_contacts=False, w_last=True))
    o = torch.stack(obs, dim=1)
    return o.view(n, -1)
