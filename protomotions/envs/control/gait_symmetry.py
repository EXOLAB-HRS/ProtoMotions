"""Left-right gait symmetry for the VIC teacher (vic_tc_v1 candidate option, 2026-10-28).

Why: vic_TC_s5_r15 walks with asymmetric arms (shoulder flexion mean L 15 / R 40 deg, swing range 19 / 33 deg; right
elbow flexed 9 vs left 26 deg; the motion pack is symmetric at +-27 deg) and stance time slightly outside the human
symmetry p95 (3.8-4.0 vs 3.7). The s5 discriminator sees no arm bodies, so nothing pulls the arms toward either side.

VicSymSteering keeps, per env and per leg, a running table of the paired joints' angles over that leg's gait percent
(sym_bins bins, exponential average sym_ema per visit). Each control step compares each side's current angles with the
other side's table at the same gait percent:

    e_j = (s_j * q_side,j - T_other,j(pct_side)) / sigma_j,   r_side = exp(-0.5 * mean_j e_j^2)
    gait_symmetry = mean of the two sides (1 where the gait percent is not valid or the other side's bin is unseen)

s_j = -1 for joints whose left / right axes are mirrored (checked on the motion pack: Shoulder_x, Shoulder_z, Elbow_z,
Thorax_x, Thorax_z have opposite-sign means; sagittal leg joints, Shoulder_y and Thorax_y the same sign). Arms are
indexed by the gait percent of the same-side leg, so each arm is compared with the other arm at the same point of its
own leg's cycle. Tables reset with the env.
"""
from dataclasses import dataclass

import torch
from torch import Tensor

from protomotions.envs.control.cadence_gait_prior import steering_path
from protomotions.envs.control.vic_steering import VicSteering, VicSteeringConfig

GAIT_SYMMETRY = steering_path("gait_symmetry")


def gait_symmetry_reward(value: Tensor) -> Tensor:
    """Reward = value (in [0, 1], 1 = symmetric)."""
    return value


@dataclass
class VicSymSteeringConfig(VicSteeringConfig):
    _target_: str = "protomotions.envs.control.gait_symmetry.VicSymSteering"
    # (joint without side prefix, sign of the right side relative to the left, sigma deg)
    sym_joints: tuple = (("Hip_y", 1.0, 5.0), ("Knee_y", 1.0, 5.0), ("Ankle_y", 1.0, 5.0),
                         ("Shoulder_x", -1.0, 6.0), ("Shoulder_y", 1.0, 6.0), ("Shoulder_z", -1.0, 6.0),
                         ("Elbow_z", -1.0, 6.0))
    sym_bins: int = 50
    sym_ema: float = 0.1


class VicSymSteering(VicSteering):
    def __init__(self, config, env):
        super().__init__(config, env)
        c = config
        names = list(env.robot_config.kinematic_info.dof_names)
        dev = self._tar_speed.device
        n = len(self._tar_speed)
        self._sym_idx = torch.tensor([[names.index(f"{s}_{j}") for j, _, _ in c.sym_joints] for s in "LR"], device=dev)
        self._sym_sign = torch.tensor([[1.0] * len(c.sym_joints), [sg for _, sg, _ in c.sym_joints]], device=dev)
        self._sym_sigma = torch.tensor([sd for _, _, sd in c.sym_joints], device=dev)
        self._sym_table = torch.zeros(2, n, c.sym_bins, len(c.sym_joints), device=dev)
        self._sym_seen = torch.zeros(2, n, c.sym_bins, dtype=torch.bool, device=dev)
        self._sym = torch.ones(n, device=dev)

    def reset(self, env_ids):
        super().reset(env_ids)
        if len(env_ids) and hasattr(self, "_sym_table"):
            self._sym_table[:, env_ids] = 0.0
            self._sym_seen[:, env_ids] = False
            self._sym[env_ids] = 1.0

    def _update_symmetry(self, pct: Tensor, valid: Tensor):
        c = self.config
        sim = self.env.simulator
        q = torch.rad2deg(sim._get_simulator_dof_state().convert_to_common(sim.data_conversion).dof_pos)
        n = q.shape[0]
        ar = torch.arange(n, device=q.device)
        b = (pct.clamp(0, 1) * (c.sym_bins - 1)).round().long()                      # [n, 2]
        rs = []
        for side in (0, 1):
            other = 1 - side
            x = q[:, self._sym_idx[side]] * self._sym_sign[side]                       # [n, J], left-side convention
            ref = self._sym_table[other, ar, b[:, side]]
            seen = self._sym_seen[other, ar, b[:, side]]
            e = (x - ref) / self._sym_sigma
            r = torch.exp(-0.5 * (e ** 2).mean(-1))
            rs.append(torch.where(valid[:, side] & seen, r, torch.ones_like(r)))
            # update this side's table at its bin
            cur = self._sym_table[side, ar, b[:, side]]
            was = self._sym_seen[side, ar, b[:, side]]
            new = torch.where(was[:, None], cur + c.sym_ema * (x - cur), x)
            upd = valid[:, side]
            self._sym_table[side, ar[upd], b[upd, side]] = new[upd]
            self._sym_seen[side, ar[upd], b[upd, side]] = True
        self._sym = (rs[0] + rs[1]) / 2

    def populate_context(self, ctx):
        super().populate_context(ctx)
        self._update_symmetry(ctx.steering.gait_pct, ctx.steering.gait_pct_valid)
        ctx.steering.gait_symmetry = self._sym
