"""teacher_se_v1 privileged vestibular observations (SE_v1 I/O map, MC teacher row, 2026-10-24).

Both are true values (no delay or noise); the student gets the sensor-model versions.
- pelvis gravity direction: unit gravity vector in the pelvis frame (3).
- head specific force: f = R_head^T (a_head - g) in m/s^2 (3), what the otoliths sense. a_head is the
  finite difference of the head's linear velocity over one control step (history slot 0 = t - dt), so
  it is 0 on the first step after a reset (the history buffer repeats the reset state) and f = -R^T g.
"""
import torch
from torch import Tensor

from protomotions.utils import rotations

GRAVITY = 9.81


def compute_pelvis_gravity_obs(root_rot: Tensor, w_last: bool) -> Tensor:
    down = torch.zeros(root_rot.shape[0], 3, device=root_rot.device, dtype=root_rot.dtype)
    down[:, 2] = -1.0
    return rotations.quat_rotate_inverse(root_rot, down, w_last)


def compute_head_specific_force_obs(body_rot: Tensor, body_vel: Tensor, historical_body_vel: Tensor,
                                    dt: float, head_idx: int, w_last: bool) -> Tensor:
    accel = (body_vel[:, head_idx] - historical_body_vel[:, 0, head_idx]) / dt
    accel[:, 2] += GRAVITY          # a - g with g = (0, 0, -9.81)
    return rotations.quat_rotate_inverse(body_rot[:, head_idx], accel, w_last)
