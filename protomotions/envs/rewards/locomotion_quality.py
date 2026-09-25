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


def reference_bounded_head_motion(
    body_rot: Tensor,
    body_ang_vel: Tensor,
    head_index: int,
    chest_index: int,
    tilt_limit_rad: float,
    relative_speed_limit_rad_s: float,
) -> Tensor:
    """Reward head motion inside a measured reference envelope.

    Shared world rotation of the head and chest cancels in the relative angular
    velocity term.  Values inside both limits receive full credit; only excess
    is penalized.  Limits are experiment data, not anatomical constants.
    """
    q = body_rot[:, head_index]
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    up_z = (1 - 2 * (q[:, 0].square() + q[:, 1].square())).clamp(-1, 1)
    tilt = torch.acos(up_z)
    relative_speed = (
        body_ang_vel[:, head_index] - body_ang_vel[:, chest_index]
    ).norm(dim=-1)
    tilt_excess = torch.relu(tilt / tilt_limit_rad - 1)
    speed_excess = torch.relu(relative_speed / relative_speed_limit_rad_s - 1)
    return torch.exp(-(tilt_excess.square() + speed_excess.square()))


def target_second_difference(current: Tensor, history: Tensor,
                             scale: float = 0.05) -> Tensor:
    """Bounded PD-target second-difference reward, radians per 30-Hz step².

    history excludes current: history[:,0] is t-1, history[:,1] is t-2.
    This is a command smoothness proxy, not measured physical jerk.
    The component uses the environment's reset grace period.
    """
    delta2 = current - 2 * history[:, 0] + history[:, 1]
    return torch.exp(-delta2.square().mean(-1) / scale**2)


def head_roll_stability(
    body_rot: Tensor,
    body_ang_vel: Tensor,
    head_index: int,
    root_index: int,
    roll_scale_rad: float,
    roll_rate_scale_rad_s: float,
) -> Tensor:
    """Penalize head roll angle and roll rate, read in the root heading frame.

    Owner: b2-a7-headroll / a7-headroll. On a7 the head's excess over the
    reference is lateral: roll rate ~3x and a speed-dependent standing tilt
    (+4.5 deg at 0.8 m/s, -10.2 deg at 1.2 m/s), while pitch matches the
    reference.  Terms that also weigh pitch, or that give full credit inside a
    p95 envelope, leave most of that motion unpenalized, so this reads only the
    roll axis and pulls continuously toward zero.

    The heading frame comes from the root's forward axis projected on the
    ground, so steering yaw is not penalized.  xyzw quaternions.
    """
    qr = body_rot[:, root_index]
    qr = qr / qr.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x, y, z, w = qr.unbind(-1)
    fwd = torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y + w * z)), dim=-1)
    fwd = fwd / fwd.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    left = torch.stack((-fwd[:, 1], fwd[:, 0]), dim=-1)

    qh = body_rot[:, head_index]
    qh = qh / qh.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x, y, z, w = qh.unbind(-1)
    up = torch.stack((2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)), dim=-1)
    roll = torch.atan2((up[:, :2] * left).sum(-1), up[:, 2])
    roll_rate = (body_ang_vel[:, head_index, :2] * fwd).sum(-1)
    return torch.exp(-((roll / roll_scale_rad).square() + (roll_rate / roll_rate_scale_rad_s).square()))


def pelvis_yaw_stability(root_rot: Tensor, tar_face_dir: Tensor, yaw_scale_rad: float) -> Tensor:
    """Penalize pelvis yaw away from the commanded facing direction.

    Owner: method 1 Stage B (b2 ... -pyaw).  Every a7-lineage policy fails the
    Stage B 1.4 m/s hold on pelvis yaw (8.4 deg mean |error| against the 7.6 deg
    reference maximum), almost all of it stride-to-stride sway rather than a
    heading offset.  The heading reward's facing term is clamp(cos(error)),
    which charges 0.3% of the heading reward for that sway, so nothing opposes
    it.  xyzw quaternions; the yaw is read from the root forward axis projected
    on the ground, the same angle the gate scores.
    """
    q = root_rot / root_rot.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x, y, z, w = q.unbind(-1)
    fwd_x, fwd_y = 1 - 2 * (y * y + z * z), 2 * (x * y + w * z)
    err = torch.atan2(fwd_y * tar_face_dir[:, 0] - fwd_x * tar_face_dir[:, 1],
                      fwd_x * tar_face_dir[:, 0] + fwd_y * tar_face_dir[:, 1])
    return torch.exp(-(err / yaw_scale_rad).square())


def speed_transition_tracking(root_pos, prev_root_pos, tar_dir, tar_speed, speed_transition, dt, vel_err_scale):
    """Speed tracking scored only while the speed command changes and just after.

    Outside a transition the reward is 1, so the term adds no gradient and no bias
    there; inside it is exp(-vel_err_scale * err^2) with err the commanded speed
    minus the root speed along the commanded direction, as in the heading reward.
    """
    speed = ((root_pos - prev_root_pos)[..., :2] / dt * tar_dir).sum(-1)
    tracked = torch.exp(-vel_err_scale * (tar_speed - speed).square())
    return torch.where(speed_transition, tracked, torch.ones_like(tracked))
