"""Frontal-plane upper-body sway for the VIC teacher (candidate option, 2026-11-24).

User 2026-11-24: "정면에서 봤을때 좌우로 많이 흔들리는거 잡기 (티쳐)". Measured on the workshop stack (hmv_eval_kit_v2
metrics/sway_arm.py, steady 1.0 m/s, BS validation 11-17): head left-right excursion 14.4 cm (teacher) / 14.1 (MC+CB) / 12.6
(+BS) against pelvis 6.5 / 5.7 / 5.8 cm, head / pelvis 2.2-2.5. In human walking the head is stabilised and its lateral
motion is not larger than the pelvis' (Menz, Lord & Fitzpatrick 2003 Gait Posture 18:35: lateral head accelerations
attenuated relative to the pelvis; no human head / pelvis excursion table is in the repository -> tolerance is an
ASSUMPTION). With the head ~0.6 m above the pelvis, the teacher's extra ~8 cm peak-to-peak is about +-4 deg of head-over-
pelvis lean; human-like (ratio ~1) is within about +-1-2 deg.

    lean = atan2(lateral offset of the head from the pelvis along the pelvis-heading left axis, vertical offset)
    lateral_sway = exp(-0.5 * (relu(|lean| - sway_tol_deg) / sway_sigma_deg)^2)        (1 inside the dead zone)

VicSymSwaySteering = VicSymSteering (gait symmetry, armsym1 recipe) + this term; reward weight from the recipe.

sway_mode "head_vel" (2026-11-25): the lean term (default "lean") let the policy satisfy it by moving the pelvis under the
head (run G: head 13.0 -> 12.3 cm, pelvis 5.0 -> 6.9 cm, W2 fell). "head_vel" scores the head's own lateral velocity in the
world, projected on the pelvis-heading left axis, with a dead zone:
    lateral_sway = exp(-0.5 * (relu(|v_head_lat| - sway_vel_tol) / sway_vel_sigma)^2)
Tolerance 0.15 m/s (ASSUMPTION, no human table in the repository): a head moving like a human pelvis (4-5 cm peak-to-peak at
the ~0.9 Hz stride frequency) peaks near pi * 0.9 * 0.05 = 0.14 m/s; the workshop teacher's 13 cm gives ~0.37 m/s.
"""
from dataclasses import dataclass

import math
import torch
from torch import Tensor

from protomotions.envs.control.cadence_gait_prior import pelvis_frame, steering_path
from protomotions.envs.control.gait_symmetry import VicSymSteering, VicSymSteeringConfig
from protomotions.envs.control.vic_steering import fill_missing_fields

LATERAL_SWAY = steering_path("lateral_sway")


def head_lateral_lean(body_pos: Tensor, pelvis_idx: int, head_idx: int, left: Tensor) -> Tensor:
    """Head-over-pelvis lean in the frontal plane (rad, + = toward the pelvis' left). body_pos [N, B, 3], left [N, 3]."""
    v = body_pos[:, head_idx] - body_pos[:, pelvis_idx]
    lat = (v * left).sum(-1)
    return torch.atan2(lat, v[:, 2].clamp_min(1e-3))


def lateral_sway_value(lean: Tensor, tol_deg: float, sigma_deg: float) -> Tensor:
    e = torch.relu(lean.abs() - math.radians(tol_deg)) / math.radians(sigma_deg)
    return torch.exp(-0.5 * e * e)


def lateral_sway_reward(value: Tensor) -> Tensor:
    """Reward = value (in [0, 1], 1 = head over the pelvis within the tolerance)."""
    return value


@dataclass
class VicSymSwaySteeringConfig(VicSymSteeringConfig):
    _target_: str = "protomotions.envs.control.lateral_sway.VicSymSwaySteering"
    sway_tol_deg: float = 2.0
    sway_sigma_deg: float = 2.0
    sway_pelvis_body: str = "Pelvis"
    sway_head_body: str = "Head"
    sway_mode: str = "lean"              # "lean" (head-over-pelvis angle) | "head_vel" (head lateral velocity, world)
    sway_vel_tol: float = 0.15           # m/s, head_vel mode
    sway_vel_sigma: float = 0.10         # m/s, head_vel mode


class VicSymSwaySteering(VicSymSteering):
    def __init__(self, config, env):
        fill_missing_fields(config)
        super().__init__(config, env)
        names = list(env.robot_config.kinematic_info.body_names)
        self._sway_idx = (names.index(config.sway_pelvis_body), names.index(config.sway_head_body))
        self._sway = torch.ones(len(self._tar_speed), device=self._tar_speed.device)
        self.head_lean = torch.zeros_like(self._sway)

    def populate_context(self, ctx):
        super().populate_context(ctx)
        c = self.config
        sim = self.env.simulator
        _, _, left = pelvis_frame(sim.get_root_state().root_rot)
        st = sim.get_robot_state()
        self.head_lean = head_lateral_lean(st.rigid_body_pos, *self._sway_idx, left)
        if c.sway_mode == "head_vel":
            v = (st.rigid_body_vel[:, self._sway_idx[1]] * left).sum(-1)
            self.head_lat_vel = v
            e = torch.relu(v.abs() - c.sway_vel_tol) / c.sway_vel_sigma
            self._sway = torch.exp(-0.5 * e * e)
        else:
            self._sway = lateral_sway_value(self.head_lean, c.sway_tol_deg, c.sway_sigma_deg)
        ctx.steering.lateral_sway = self._sway
