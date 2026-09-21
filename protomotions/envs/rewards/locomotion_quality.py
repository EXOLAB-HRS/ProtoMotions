"""Candidate reward kernels owned by a9/a10 (a7 follow-up experiments)."""
import torch
from torch import Tensor


def head_upright(body_rot: Tensor, body_index: int, tilt_scale: float = 0.174533) -> Tensor:
    """World head-up tilt; xyzw, yaw-invariant, rejects upside-down heads too."""
    q = body_rot[:, body_index]
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    up_z = (1 - 2 * (q[:, 0].square() + q[:, 1].square())).clamp(-1, 1)
    # Squared chord distance is well-conditioned at zero tilt, unlike acos.
    return torch.exp(-2 * (1 - up_z) / (tilt_scale * tilt_scale))


def head_angular_stability(body_ang_vel: Tensor, body_index: int,
                           speed_scale: float = 0.8) -> Tensor:
    """Suppress world pitch/roll angular velocity; leave yaw free for steering."""
    return torch.exp(-body_ang_vel[:, body_index, :2].square().sum(-1) / speed_scale**2)


def target_second_difference(current: Tensor, history: Tensor,
                             scale: float = 0.05) -> Tensor:
    """Bounded PD-target second-difference reward, radians per 30-Hz step².

    history excludes current: history[:,0] is t-1, history[:,1] is t-2.
    This is a command smoothness proxy, not measured physical jerk.
    The component uses the environment's reset grace period.
    """
    delta2 = current - 2 * history[:, 0] + history[:, 1]
    return torch.exp(-delta2.square().mean(-1) / scale**2)
