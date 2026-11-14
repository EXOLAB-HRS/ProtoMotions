# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""Joint-level series elastic element (SEE) of the ankle: Achilles (plantarflexion) and tibialis anterior (dorsiflexion).

human_model_v4.0 (candidate, 2026-11-14, user: "joint level plant에 tendon넣은거를 v4.0이라고 하고 일단 모델은 만들어두자").
Off unless human_model_parameters["series_elastic"] is set (human_model_v4_0/model_config.py).

Per SEE joint (L/R Ankle_y; plant sign + = plantarflexion) there are two antagonist channels c (PF: s = +1, DF: s = -1),
each a massless contractile element (CE) with a hidden CE angle phi_c in series with a tension-only tendon:

    D_c   = s (phi_c - q)                               tendon stretch in joint-angle units (rad), D = delta / r
    T(D)  = 0                    D <= 0                 tendon force as joint torque (N m): quadratic toe region,
          = k D^2 / (2 D_toe)    0 < D < D_toe          then linear (C1 at D_toe); k = k_tendon * r^2
          = k (D - D_toe / 2)    D >= D_toe
    u(x, xd) = Kp (q_tgt - x) - Kd xd + ff               the policy's (VIC) active law, now evaluated at the CE
    a_c   = clip(s u(phi_c, phid_c), 0, cap_c)           CE torque, tension only, strength cap of channel c
    a_c   = T(D_c) + c_T Dd_c                            massless CE: CE force = tendon force (+ tendon damping c_T)
    tau_joint = sum_c s a_c                              applied to the joint as an explicit effort

phi_c is advanced by backward Euler (phid = (phi - phi_prev) / dt) with q held at the substep start; the scalar
equation is monotone in D and is solved in closed form (piecewise: unsaturated / at cap / CE silent), so the step is
unconditionally stable on the CE side and graph-capturable (no data-dependent control flow). The joint only sees the
tendon (stiffness <= k) as an explicit torque. A silent CE (u <= 0 for its channel, no tendon damping) gives D = 0,
i.e. the CE follows the joint with the tendon just taut at zero force (no slack take-up delay).

Strength caps (Anderson 2007 Hill-type, active_strength_scale of v3.6) are evaluated at the joint angle and at the
CE velocity of the engaged channel (previous substep), not the joint velocity: an isokinetic dynamometer measures the
torque-velocity curve at constant torque, where tendon length is constant and CE velocity = joint velocity; in walking
push-off the tendon recoils and the CE is much slower than the joint (Hof 2002, Fukunaga 2001, Lichtwark & Wilson 2006).
This keeps the idea of the 2026-11-06 audit draft (share/human_model_audit_261106 tendon_v4_0.py: cap at
qd_ce = qd + tau_dot / k_T) but takes qd_ce from the integrated CE state instead of a low-passed torque derivative.

Limit k -> inf (D_toe = 0, c_T = 0): phi -> q and tau_joint -> clip(u(q, qd), -N, P), the v3.6 active torque.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class TendonChannel:
    """One tension-only SEE channel in joint units. sign +1 = plantarflexor (Achilles), -1 = dorsiflexor (TA)."""
    sign: float
    k_nm_per_rad: float          # linear-region rotational stiffness k_tendon * r^2
    toe_rad: float               # end of the toe region in stretch angle (eps_toe * L_slack / r)
    damping_nms_per_rad: float   # tendon damping c_T (hysteresis stand-in)
    moment_arm_m: float          # only for reporting (strain, elongation, force)
    slack_length_m: float        # only for reporting (strain)

    def validate(self):
        if self.sign not in (1.0, -1.0):
            raise ValueError("tendon channel sign must be +1 or -1")
        for name in ("k_nm_per_rad", "moment_arm_m", "slack_length_m"):
            v = getattr(self, name)
            if not (math.isfinite(v) and v > 0):
                raise ValueError(f"tendon {name} must be positive")
        for name in ("toe_rad", "damping_nms_per_rad"):
            v = getattr(self, name)
            if not (math.isfinite(v) and v >= 0):
                raise ValueError(f"tendon {name} must be >= 0")


def tendon_torque(D: Tensor, k, toe) -> Tensor:
    """T(D) >= 0 (N m) for stretch D (rad); k, toe broadcastable."""
    Dp = D.clamp_min(0)
    toe_t = max(float(toe), 1e-12)   # python scalar (CUDA-graph capture safe)
    quad = k * Dp * Dp / (2 * toe_t)
    lin = k * (Dp - toe_t / 2)
    return torch.where(Dp < toe_t, quad, lin)


def tendon_energy(D: Tensor, k, toe) -> Tensor:
    """Elastic energy (J) stored at stretch D: integral of tendon_torque."""
    Dp = D.clamp_min(0)
    toe_t = max(float(toe), 1e-12)   # python scalar (CUDA-graph capture safe)
    cubic = k * Dp ** 3 / (6 * toe_t)
    lin = k * toe_t ** 2 / 6 + k * ((Dp * Dp - toe_t * toe_t) / 2 - toe_t * (Dp - toe_t) / 2)
    return torch.where(Dp < toe_t, cubic, lin)


def tendon_stretch(torque: Tensor, k, toe) -> Tensor:
    """Inverse of tendon_torque for torque >= 0."""
    F = torque.clamp_min(0)
    toe_t = max(float(toe), 1e-12)   # python scalar (CUDA-graph capture safe)
    f_toe = k * toe_t / 2
    return torch.where(F < f_toe, torch.sqrt(2 * F * toe_t / k), F / k + toe_t / 2)


def _solve(c: Tensor, a: Tensor, k, toe) -> Tensor:
    """Root D >= 0 of c - a D = T(D) (a >= 0); D = 0 when c <= 0. Closed form per branch."""
    toe_t = max(float(toe), 1e-12)   # python scalar: a host->device tensor copy is not allowed inside CUDA-graph capture
    cp = c.clamp_min(0)
    lin = (cp + k * toe_t / 2) / (a + k)
    disc = torch.sqrt(a * a + 2 * k * cp / toe_t)
    quad = 2 * cp / (a + disc).clamp_min(1e-12)
    g_toe = cp - a * toe_t - k * toe_t / 2            # sign of c - aD - T(D) at D = toe
    return torch.where(g_toe >= 0, lin, quad)


def channel_step(c: Tensor, a1: Tensor, cap: Tensor, D_prev: Tensor, dt: float, ch: TendonChannel):
    """One backward-Euler step of a channel. c = s u(phi = q, phid = (q - phi_prev)/dt); a1 = Kp + Kd / dt.

    Solves clip(c - a1 D, 0, cap) = T(D) + c_T (D - D_prev) / dt for D >= 0 (left side non-increasing, right side
    increasing in D: one root). Returns (D, CE torque a >= 0 = tendon force as joint torque magnitude).
    """
    k, toe, cT = ch.k_nm_per_rad, ch.toe_rad, ch.damping_nms_per_rad
    h = cT / dt
    zero = torch.zeros_like(c)
    D_free = _solve(c + h * D_prev, a1 + h, k, toe)              # CE between 0 and cap
    D_cap = _solve(cap + h * D_prev, zero + h, k, toe)           # CE at its cap
    D_off = _solve(h * D_prev, zero + h, k, toe)                 # CE silent (tendon damping only)
    use_cap = (c - a1 * D_cap) >= cap
    use_off = (c - a1 * D_off) <= 0
    D = torch.where(use_cap, D_cap, torch.where(use_off, D_off, D_free))
    a = (c - a1 * D).clamp(min=0).minimum(cap.clamp_min(0))
    return D, a


class SeriesElasticAnkle:
    """Stateful SEE for a set of joints; buffers are updated in place (CUDA-graph safe).

    joints: DOF names (plant order index resolved against dof_names). channels: {"PF": TendonChannel, "DF": ...}.
    """

    def __init__(self, config: Dict, dof_names: Sequence[str], num_envs: int, device, dtype=torch.float32):
        self.config = config
        self.joints = list(config["joints"])
        self.idx = torch.tensor([list(dof_names).index(j) for j in self.joints], device=device, dtype=torch.long)
        self.channels = {name: TendonChannel(**config["channels"][name]) for name in ("PF", "DF")}
        for ch in self.channels.values():
            ch.validate()
        if self.channels["PF"].sign != 1.0 or self.channels["DF"].sign != -1.0:
            raise ValueError("PF channel sign must be +1 and DF channel sign -1 (plant: + plantarflexion)")
        self.cap_at_ce_velocity = bool(config.get("cap_at_ce_velocity", True))
        n, j = num_envs, len(self.joints)
        z = lambda: torch.zeros((n, j), device=device, dtype=dtype)
        self.phi = {c: z() for c in self.channels}        # CE angle per channel (rad, joint frame)
        self.phid = {c: z() for c in self.channels}       # CE velocity of the last substep (rad/s)
        self.D = {c: z() for c in self.channels}          # tendon stretch (rad) of the last substep
        self.force = {c: z() for c in self.channels}      # CE = tendon torque magnitude (N m)
        self.fresh = torch.ones((n, 1), device=device, dtype=torch.bool)
        self.mask = torch.zeros(len(dof_names), device=device, dtype=torch.bool)
        self.mask[self.idx] = True
        # Full-width diagnostics for rewards / recorders (zeros outside the SEE joints).
        self.ce_power_full = torch.zeros((n, len(dof_names)), device=device, dtype=dtype)
        self.tendon_energy = z()

    # ---- state ----
    def reset(self, env_ids=None):
        if env_ids is None:
            self.fresh.fill_(True)
        else:
            self.fresh[env_ids] = True

    def state_dict(self):
        return {f"{k}_{c}": v[c].clone() for k, v in (("phi", self.phi), ("phid", self.phid), ("D", self.D),
                                                      ("force", self.force)) for c in self.channels} | {
            "fresh": self.fresh.clone()}

    def load_state_dict(self, s):
        for k, v in (("phi", self.phi), ("phid", self.phid), ("D", self.D), ("force", self.force)):
            for c in self.channels:
                v[c].copy_(s[f"{k}_{c}"])
        self.fresh.copy_(s["fresh"])

    def cap_velocity(self, qd: Tensor) -> Tensor:
        """Joint velocities with SEE columns replaced by the engaged channel's CE velocity (for strength caps)."""
        if not self.cap_at_ce_velocity:
            return qd
        qj = qd[:, self.idx]
        pf_on, df_on = self.force["PF"] > 0, self.force["DF"] > 0
        ce = torch.where(pf_on, self.phid["PF"], torch.where(df_on, self.phid["DF"], qj))
        ce = torch.where(self.fresh, qj, ce)
        out = qd.clone()
        out[:, self.idx] = ce
        return out

    # ---- one physics substep ----
    def substep(self, q: Tensor, qd: Tensor, target: Tensor, kp: Tensor, kd: Tensor, ff: Tensor,
                negative: Tensor, positive: Tensor, dt: float):
        """All tensors [N, ndof] (plant order). Returns (joint torque [N, nsee], tendon torque per channel dict).

        kp / kd: the CE gains (policy VIC gains, plus any beta-prior gaps); ff: state-independent feed-forward
        (VIC ff_res + reflex + static prior). negative / positive: strength caps (>= 0) for the SEE columns.
        """
        i = self.idx
        qj, qdj, tgt = q[:, i], qd[:, i], target[:, i]
        kpj, kdj, ffj = kp[:, i], kd[:, i], ff[:, i]
        caps = {"PF": positive[:, i], "DF": negative[:, i]}
        a1 = kpj + kdj / dt
        tau = torch.zeros_like(qj)
        for name, ch in self.channels.items():
            s = ch.sign
            phi_prev = torch.where(self.fresh, qj, self.phi[name])
            D_prev = torch.where(self.fresh, torch.zeros_like(qj), self.D[name])
            u = kpj * (tgt - qj) - kdj * (qj - phi_prev) / dt + ffj      # u at phi = q, backward-Euler velocity
            D, a = channel_step(s * u, a1, caps[name], D_prev, dt, ch)
            phi = qj + s * D
            self.phid[name].copy_((phi - phi_prev) / dt)
            self.phi[name].copy_(phi)
            self.D[name].copy_(D)
            self.force[name].copy_(a)
            tau = tau + s * a
        self.fresh.fill_(False)
        # CE (muscle) mechanical power and tendon energy, for metabolic proxies and recorders.
        ce_power = sum(ch.sign * self.force[n] * self.phid[n] for n, ch in self.channels.items())
        self.ce_power_full[:, i] = ce_power
        self.tendon_energy.copy_(sum(tendon_energy(self.D[n], ch.k_nm_per_rad, ch.toe_rad)
                                     for n, ch in self.channels.items()))
        return tau

    def diagnostics(self):
        """Per channel stretch (rad), elongation (m), strain (-), force (N) and CE angle / velocity."""
        out = {}
        for n, ch in self.channels.items():
            out[n] = dict(D_rad=self.D[n], elongation_m=self.D[n] * ch.moment_arm_m,
                          strain=self.D[n] * ch.moment_arm_m / ch.slack_length_m,
                          force_n=self.force[n] / ch.moment_arm_m, phi=self.phi[n], phid=self.phid[n])
        return out


def build(config: Optional[Dict], dof_names, num_envs, device, dtype=torch.float32):
    if not config:
        return None
    return SeriesElasticAnkle(config, dof_names, num_envs, device, dtype)


def channel_from_tendon(sign, k_tendon_n_per_m, moment_arm_m, slack_length_m, toe_strain, damping_nms_per_rad):
    """Convert tendon-level numbers to a joint-level channel: k = k_tendon r^2, D_toe = eps_toe L / r."""
    return asdict(TendonChannel(sign=float(sign), k_nm_per_rad=float(k_tendon_n_per_m) * moment_arm_m ** 2,
                                toe_rad=float(toe_strain) * slack_length_m / moment_arm_m,
                                damping_nms_per_rad=float(damping_nms_per_rad), moment_arm_m=float(moment_arm_m),
                                slack_length_m=float(slack_length_m)))


def check_explicit_stability(see: SeriesElasticAnkle, control_info, fps: float, safety: float = 0.5):
    """The joint sees the tendon as an explicit torque: require k <= safety * I (2 fps)^2 and c_T / (I fps) <= 0.5.

    I is a lower bound of the joint-space inertia: the joint armature (0.02 kg m^2 on the human ankle; the foot adds
    ~0.01 about the ankle axis). Raises if a channel is too stiff for the physics rate.
    """
    for j in see.joints:
        arm = getattr(control_info[j], "armature", None)
        if arm is None or not arm > 0:
            raise ValueError(f"series_elastic needs a positive armature on {j} for the stability bound")
        for name, ch in see.channels.items():
            k_max = safety * arm * (2.0 * fps) ** 2
            if ch.k_nm_per_rad > k_max or ch.damping_nms_per_rad / (arm * fps) > 0.5:
                raise ValueError(f"series_elastic {name} on {j}: k {ch.k_nm_per_rad:.0f} N m/rad (limit {k_max:.0f}) / "
                                 f"c_T {ch.damping_nms_per_rad:.2f} too stiff for {fps:.0f} Hz explicit coupling")
