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

Dead-zone option (sym_deadzone, 2026-10-28, user-approved; default off = the reward above):

    e_j = relu(|s_j * q_side,j - T_other,j(pct_side)| - tol_j) / sigma_j

so a left-right difference within the human tolerance tol_j costs nothing and only the excess is penalised. tol_j =
human p95 of the per-subject mean |L - R| over the gait cycle (SYM_TOL_DEG; minimum of WBDS, Fukuchi 2018, 495 L/R
pairs and Van Criekinge 2023 able-bodied, 138 pairs). With sym_stance_tol_pct > 0 the reward is also multiplied by
exp(-0.5 (relu(|st_L - st_R| - tol) / sym_stance_sigma_pct)^2), st = stance % of the leg's last complete cycles (EMA,
sym_stance_ema), tol = 3.73 % (human p95, HMV-Walk criteria_w2v4 stance_symmetry).
"""
from dataclasses import dataclass, field

import torch
from torch import Tensor

from protomotions.envs.control.cadence_gait_prior import steering_path
from protomotions.envs.control.vic_steering import VicSteering, VicSteeringConfig, fill_missing_fields

GAIT_SYMMETRY = steering_path("gait_symmetry")

# Human left-right tolerance (deg), p95 over subjects of the mean |L - R| over the gait cycle (min of WBDS / Van Criekinge;
# output/261019/e01_bg_circuit_pathology_validation/evaluation/human/*_curves.npz): hip flexion 6.72 (VC; WBDS 15.4 holds
# pelvis-tilt offsets), knee 8.69 (VC; WBDS 9.04), ankle 5.31 (WBDS; VC 7.03), shoulder flexion 9.01, elbow 9.76 (VC).
# Shoulder_x / Shoulder_z: no human data in these sets, assumption = shoulder flexion.
SYM_TOL_DEG = {"Hip_y": 6.7, "Knee_y": 8.7, "Ankle_y": 5.3, "Shoulder_x": 9.0, "Shoulder_y": 9.0, "Shoulder_z": 9.0,
               "Elbow_z": 9.8}


def symmetry_error(x: Tensor, ref: Tensor, sigma: Tensor, tol: Tensor = None) -> Tensor:
    """Normalised left-right error per joint; with tol, zero inside the dead zone |x - ref| <= tol."""
    d = x - ref
    if tol is not None:
        d = torch.relu(d.abs() - tol)
    return d / sigma


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
    sym_deadzone: bool = False           # 2026-10-28: zero penalty inside the human tolerance (SYM_TOL_DEG)
    sym_tol_deg: dict = field(default_factory=dict)   # overrides of SYM_TOL_DEG by joint
    sym_stance_tol_pct: float = 0.0      # > 0: stance % symmetry factor with this dead zone (3.73 = human p95)
    sym_stance_sigma_pct: float = 2.0
    sym_stance_ema: float = 0.3


def stance_symmetry_factor(frac: Tensor, ok: Tensor, tol: float, sigma: float) -> Tensor:
    """exp(-0.5 (relu(|L - R| - tol) / sigma)^2) where both legs have a stance %, else 1."""
    e = torch.relu((frac[:, 0] - frac[:, 1]).abs() - tol) / sigma
    return torch.where(ok, torch.exp(-0.5 * e ** 2), torch.ones_like(e))


class VicSymSteering(VicSteering):
    def __init__(self, config, env):
        fill_missing_fields(config)
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
        tol = dict(SYM_TOL_DEG, **c.sym_tol_deg)
        self._sym_tol = torch.tensor([tol[j] for j, _, _ in c.sym_joints], device=dev) if c.sym_deadzone else None
        self._st_cnt = torch.zeros(n, 2, 2, device=dev)              # steps in [stance, swing] of the current cycle
        self._st_frac = torch.zeros(n, 2, device=dev)                # EMA stance fraction of complete cycles (%)
        self._st_seen = torch.zeros(n, 2, dtype=torch.bool, device=dev)
        self._st_prev = torch.zeros(n, 2, dtype=torch.bool, device=dev)
        self._st_cycle = torch.zeros(n, 2, dtype=torch.bool, device=dev)   # a stance onset has been seen

    def reset(self, env_ids):
        super().reset(env_ids)
        if len(env_ids) and hasattr(self, "_sym_table"):
            self._sym_table[:, env_ids] = 0.0
            self._sym_seen[:, env_ids] = False
            self._sym[env_ids] = 1.0
            self._st_cnt[env_ids] = 0.0
            self._st_frac[env_ids] = 0.0
            self._st_seen[env_ids] = False
            self._st_prev[env_ids] = False
            self._st_cycle[env_ids] = False

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
            e = symmetry_error(x, ref, self._sym_sigma, self._sym_tol)
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
        if c.sym_stance_tol_pct > 0:
            self._sym = self._sym * self._stance_symmetry(valid)

    def _stance_symmetry(self, valid: Tensor) -> Tensor:
        """Stance % of each leg's complete cycles (onset to onset, EMA) and the dead-zone factor on |L - R|."""
        c = self.config
        st = self._stance
        onset = st & ~self._st_prev
        self._st_prev = st.clone()
        done = onset & self._st_cycle
        tot = self._st_cnt.sum(-1).clamp_min(1.0)
        frac = 100.0 * self._st_cnt[..., 0] / tot
        new = torch.where(self._st_seen, self._st_frac + c.sym_stance_ema * (frac - self._st_frac), frac)
        self._st_frac = torch.where(done, new, self._st_frac)
        self._st_seen |= done
        self._st_cnt = torch.where(onset[..., None], torch.zeros_like(self._st_cnt), self._st_cnt)
        self._st_cycle |= onset
        self._st_cnt[..., 0] += st.float()
        self._st_cnt[..., 1] += (~st).float()
        return stance_symmetry_factor(self._st_frac, self._st_seen.all(-1) & valid.all(-1),
                                      c.sym_stance_tol_pct, c.sym_stance_sigma_pct)

    def populate_context(self, ctx):
        super().populate_context(ctx)
        self._update_symmetry(ctx.steering.gait_pct, ctx.steering.gait_pct_valid)
        ctx.steering.gait_symmetry = self._sym
