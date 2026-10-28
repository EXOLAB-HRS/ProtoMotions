# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""Joint-level spinal stretch reflex on the plant side (candidate, 2026-10-28, default off).

Stand-in for the per-muscle stretch reflex of the muscle-level plant (human-plant-v2
``control/stretch_reflex.py`` + ``reflex_delay.py``), with the same structure so that one can replace
the other: a sensor (here joint deviation from the PD target instead of MTU length / velocity), a
discrete physics-rate delay with zero prehistory, and rectified phasic / tonic laws per antagonist
group. Each selected joint has two groups (index 0: the group stretched when the joint deviation
velocity is negative, index 1: positive); a stretched group makes torque against its stretch.

Per physics substep, per env, joint j and group s (sign sigma_s = -1 / +1):

    e   = q - q_ref                      deviation from the held PD target (rad)
    e_v = qdot - qdot_ref                deviation velocity; qdot_ref = (q_ref[k] - q_ref[k-1]) / dt_ctrl,
                                         held over control step k (first step after a reset: qdot_ref = qdot)
    x(t - d)                             both read from a ring buffer, d ~ U(delay_ms) per env at reset
    stretch_s = sigma_s * e_v(t-d) > 0   and, with away_only, sigma_s * e(t-d) > 0 (deviation growing)
    tau_s = -sigma_s * g * [ k_v,s (sigma_s e_v - v_th,s)_+  +  k_l,s (sigma_s e - l_th,s)_+ ]
    g     = gain_scale * (1 + kappa * u_+)      u = VIC co-activation level of the joint (held per control step)
    tau_r = clip(tau_0 + tau_1, +- max_frac * effort_limit)

tau_r is added to the VIC feed-forward torque inside the implicit-PD substep (dynamics._vic_feedforward), so
feed-forward + reflex is clipped to the strength caps and the PhysX drive keeps the remaining headroom.
The tonic term is off by default (k_l = 0). With gain_scale = 0 the block is a pure sensor (calibration).

Not a physiological reflex model: the sensor is the deviation from the controller's own target, thresholds are
set from the simulation (silent in steady walking), and the gains are order-of-magnitude choices
(see JointReflexConfig). Delays, buffer and gating follow human short-latency (M1) stretch reflexes.
"""
from dataclasses import dataclass, field
import atexit
import json
import math
import os
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor

VIC_JOINTS = ("L_Hip_y", "L_Knee_y", "L_Ankle_y", "R_Hip_y", "R_Knee_y", "R_Ankle_y",
              "L_Hip_x", "R_Hip_x", "L_Subtalar_x", "R_Subtalar_x")

# k_v = c_s * K_stance with c_s = 0.032 s, K_stance = geometric centre of the human stance stiffness range used by
# vic_steering (HUMAN_RANGES / HUMAN_RANGES_FRONTAL, N m/rad): Hip_y 126, Knee_y 33.5, Ankle_y 94.5, Hip_x 265,
# Subtalar_x 26.5. c_s is set so that the ankle gets k_v = 3 N m s/rad: reflex ankle stiffness about 30 % of the
# intrinsic stance stiffness at a ~1.5 Hz perturbation (k_v * omega ~ 0.3 K at omega ~ 9.4 rad/s; reflex share of
# ankle stiffness 20-40 % in standing / stance, Sinkjaer et al. 1988, 1996; Kearney et al. 1997; Mirbagheri et al.
# 2000), and a 200 deg/s ankle perturbation gives ~6-12 N m above a 1.5 rad/s threshold (magnitude of reported
# short-latency reflex torques, a few to ~20 N m). Other joints: same c_s (uncertain; no comparable data).
DEFAULT_KV = {"Hip_y": 4.0, "Knee_y": 1.1, "Ankle_y": 3.0, "Hip_x": 8.5, "Subtalar_x": 0.85}
# Placeholder threshold (rad/s), uncalibrated: must be replaced by >= p99 of |e_v| in steady walking per joint
# (calibration run with gain_scale = 0, calibrate = True; HC_JOINT_REFLEX_CALIBRATE).
DEFAULT_VTH = {"Hip_y": 1.5, "Knee_y": 1.5, "Ankle_y": 1.5, "Hip_x": 1.5, "Subtalar_x": 1.5}

Pair = Union[float, Tuple[float, float]]


@dataclass
class JointReflexConfig:
    enabled: bool = False
    joints: Tuple[str, ...] = VIC_JOINTS
    # Short-latency (M1, spinal) stretch reflex of human leg muscles ~25-45 ms (soleus / tibialis anterior M1
    # ~35-45 ms, quadriceps / hamstrings ~25-35 ms); drawn per env at reset, rounded to physics steps.
    delay_ms: Tuple[float, float] = (30.0, 40.0)
    fixed_delay_ms: Optional[float] = None          # evaluation: one delay for every env
    max_delay_ms: float = 60.0
    # Per joint key without side ("Hip_y" ...): a float (both groups) or (group 0, group 1).
    v_th: Dict[str, Pair] = field(default_factory=lambda: dict(DEFAULT_VTH))       # rad/s
    k_v: Dict[str, Pair] = field(default_factory=lambda: dict(DEFAULT_KV))         # N m s/rad
    l_th: Dict[str, Pair] = field(default_factory=dict)                            # rad (default 0.2)
    k_l: Dict[str, Pair] = field(default_factory=dict)                             # N m/rad (default 0 = tonic off)
    away_only: bool = True           # fire only while the deviation from q_ref grows (stretch away from the target)
    kappa: float = 1.0               # descending gain modulation by the VIC co-activation level u_+
    gain_scale: float = 1.0          # g0; 0 = sensor only
    max_frac: float = 0.3            # |tau_r| <= max_frac * effort_limit of the joint
    calibrate: bool = False          # keep |e_v| histograms per joint (calibration_quantiles())
    hist_max: float = 20.0           # rad/s, histogram range
    hist_bins: int = 2000


def _pairs(table: Dict[str, Pair], joints, default: float) -> Tensor:
    out = torch.empty(len(joints), 2)
    for i, j in enumerate(joints):
        v = table.get(j, table.get(j[2:] if j[:2] in ("L_", "R_") else j, default))
        out[i] = torch.tensor(v if isinstance(v, (tuple, list)) else (v, v), dtype=torch.float)
    return out


class JointReflex:
    """Vectorized joint stretch reflex; the env calls reset / control_step, the human model calls substep."""

    def __init__(self, config: JointReflexConfig, dof_names, effort_limits, physics_hz: float, control_dt: float,
                 num_envs: int, device, generator=None):
        c = self.config = config
        self.device = torch.device(device)
        self.generator = generator
        self.joint_names = tuple(c.joints)
        names = list(dof_names)
        missing = [j for j in c.joints if j not in names]
        if missing:
            raise ValueError(f"joint reflex: unknown joints {missing}")
        self.idx = torch.tensor([names.index(j) for j in c.joints], device=self.device)
        J = len(c.joints)
        self.n, self.J = num_envs, J
        self.physics_dt = 1.0 / float(physics_hz)
        self.control_dt = float(control_dt)
        lo, hi = c.delay_ms
        top = hi if c.fixed_delay_ms is None else c.fixed_delay_ms
        if lo < 0 or lo > hi or top > c.max_delay_ms or (c.fixed_delay_ms is not None and c.fixed_delay_ms < 0):
            raise ValueError(f"joint reflex: bad delay {c.delay_ms} / fixed {c.fixed_delay_ms} / max {c.max_delay_ms}")
        self.depth = int(round(c.max_delay_ms / 1000.0 / self.physics_dt)) + 1
        dev = self.device
        self.v_th = _pairs(c.v_th, c.joints, 1.5).to(dev)
        self.k_v = _pairs(c.k_v, c.joints, 0.0).to(dev)
        self.l_th = _pairs(c.l_th, c.joints, 0.2).to(dev)
        self.k_l = _pairs(c.k_l, c.joints, 0.0).to(dev)
        if any((t < 0).any() for t in (self.v_th, self.k_v, self.l_th, self.k_l)):
            raise ValueError("joint reflex: gains and thresholds must be nonnegative")
        self.tonic = bool((self.k_l > 0).any())
        lim = torch.as_tensor([float(effort_limits[j]) for j in c.joints], dtype=torch.float)
        self.tau_cap = (c.max_frac * lim).to(dev)
        self.sigma = torch.tensor([-1.0, 1.0], device=dev)
        self.history = torch.zeros(self.depth, num_envs, J, 2, device=dev)     # (e, e_v), oldest overwritten
        self.head = 0
        self.delay_steps = torch.zeros(num_envs, dtype=torch.long, device=dev)
        self._env = torch.arange(num_envs, device=dev)
        self.q_ref = torch.zeros(num_envs, J, device=dev)
        self.qd_ref = torch.zeros(num_envs, J, device=dev)
        self.u = torch.zeros(num_envs, J, device=dev)
        self.fresh = torch.ones(num_envs, dtype=torch.bool, device=dev)
        self.tau = torch.zeros(num_envs, J, device=dev)            # last substep
        self._sum_abs = torch.zeros(num_envs, J, device=dev)       # over the substeps of the current control step
        self._sum_active = torch.zeros(num_envs, J, device=dev)
        self._substeps = 0
        self.step_abs_tau = torch.zeros(num_envs, J, device=dev)   # mean over the last control step's substeps
        self.step_active = torch.zeros(num_envs, J, device=dev)
        self.hist = torch.zeros(J, c.hist_bins, dtype=torch.long, device=dev) if c.calibrate else None
        self.hist_away = torch.zeros_like(self.hist) if c.calibrate else None
        self._vic = None
        self._vic_map = None
        self._steps = 0
        self._sample_delays(self._env)

    # ------------------------------------------------------------------ env side
    def _sample_delays(self, env_ids: Tensor):
        c = self.config
        if c.fixed_delay_ms is not None:
            ms = torch.full((len(env_ids),), float(c.fixed_delay_ms), device=self.device)
        else:
            u = torch.rand(len(env_ids), device=self.device, generator=self.generator)
            ms = c.delay_ms[0] + (c.delay_ms[1] - c.delay_ms[0]) * u
        self.delay_steps[env_ids] = torch.round(ms / 1000.0 / self.physics_dt).long().clamp(0, self.depth - 1)

    @property
    def delay_ms(self) -> Tensor:
        return self.delay_steps.float() * (1000.0 * self.physics_dt)

    def reset(self, env_ids: Tensor):
        """Zero prehistory (silent for d steps), new delay, next control step starts qdot_ref from qdot."""
        env_ids = env_ids.to(self.device, torch.long)
        if len(env_ids) == 0:
            return
        self.history[:, env_ids] = 0.0
        self.tau[env_ids] = 0.0
        self.fresh[env_ids] = True
        self._sample_delays(env_ids)

    def attach_vic(self, component):
        """VIC steering component that supplies the co-activation level u (None: u = 0)."""
        self._vic = component
        if component is None:
            return
        vj = list(component.config.vic_joints)
        self._vic_map = torch.tensor([vj.index(j) if j in vj else -1 for j in self.joint_names], device=self.device)

    def _coactivation(self) -> Tensor:
        v = self._vic
        if v is None or self.config.kappa == 0:
            return torch.zeros(self.n, self.J, device=self.device)
        get = getattr(v, "coactivation_level", None)
        if callable(get):
            lvl = get()
        else:
            # vic_steering keeps the filtered delta (delta_tau_up_s / down_s or delta_lowpass_s) and its bound;
            # without a filter _delta stays 0 and u = 0 (gain g0).
            lvl = v._delta / v._dmax.clamp_min(1e-6)
        m = self._vic_map
        out = torch.zeros(self.n, self.J, device=self.device)
        ok = m >= 0
        if ok.any():
            out[:, ok] = lvl[:, m[ok]].to(out.dtype)
        return out

    def control_step(self, q_ref_all: Tensor, qd_all: Optional[Tensor] = None):
        """Called once per control step before physics with the PD targets [N, dof] (common order).
        qd_all = current joint velocity (used for qdot_ref on the first step after a reset)."""
        q_ref = q_ref_all[:, self.idx].to(self.q_ref.dtype)
        fd = (q_ref - self.q_ref) / self.control_dt
        first = self.fresh[:, None]
        start = qd_all[:, self.idx].to(fd.dtype) if qd_all is not None else torch.zeros_like(fd)
        self.qd_ref = torch.where(first, start, fd)
        self.q_ref = q_ref.clone()
        self.fresh[:] = False
        self.u = self._coactivation()
        if self._substeps:
            self.step_abs_tau = self._sum_abs / self._substeps
            self.step_active = self._sum_active / self._substeps
        self._sum_abs.zero_()
        self._sum_active.zero_()
        self._substeps = 0
        self._steps += 1
        dump = getattr(self, "_dump_calibration", None)
        if dump is not None and self._steps % 3000 == 0:      # the simulator may exit without atexit hooks
            dump()

    # ------------------------------------------------------------------ plant side
    def compute(self, q: Tensor, qd: Tensor) -> Tensor:
        """One physics substep from the selected joints' q, qdot [N, J]; returns tau_r [N, J]."""
        c = self.config
        e = q - self.q_ref
        e_v = qd - self.qd_ref
        self.head = (self.head + 1) % self.depth
        self.history[self.head] = torch.stack((e, e_v), -1)
        slot = (self.head - self.delay_steps) % self.depth
        delayed = self.history[slot, self._env]                       # [N, J, 2]
        ed, evd = delayed[..., 0:1], delayed[..., 1:2]                # [N, J, 1] against groups [2]
        se, sv = self.sigma * ed, self.sigma * evd                    # stretch of group s, [N, J, 2]
        phasic = self.k_v * (sv - self.v_th).clamp_min(0)
        if c.away_only:
            phasic = phasic * (se > 0)
        drive = phasic
        if self.tonic:
            drive = drive + self.k_l * (se - self.l_th).clamp_min(0)
        g = c.gain_scale * (1.0 + c.kappa * self.u.clamp_min(0))      # [N, J]
        tau = -(self.sigma * drive).sum(-1) * g
        tau = torch.maximum(torch.minimum(tau, self.tau_cap), -self.tau_cap)
        self.tau = tau
        active = (drive > 0).any(-1).float()
        self._sum_abs += tau.abs()
        self._sum_active += active
        self._substeps += 1
        if self.hist is not None:
            self._histogram(evd.squeeze(-1).abs(), (ed * evd).squeeze(-1) > 0)
        return tau

    def substep(self, simulator):
        """Human-model hook: write tau_r into simulator.vic_ff_reflex (common order, in place)."""
        if getattr(simulator, "vic_ff_prior", None) is None or os.environ.get("HC_HUMAN_IMPLICIT_PD", "0") != "1":
            raise RuntimeError("joint reflex needs the VIC implicit-PD path (vic steering + HC_HUMAN_IMPLICIT_PD=1)")
        buf = getattr(simulator, "vic_ff_reflex", None)
        if buf is None or buf.shape != simulator.vic_ff_prior.shape:
            buf = simulator.vic_ff_reflex = torch.zeros_like(simulator.vic_ff_prior)
        dof = simulator._get_simulator_dof_state().convert_to_common(simulator.data_conversion)
        tau = self.compute(dof.dof_pos[:, self.idx], dof.dof_vel[:, self.idx])
        buf[:, self.idx] = tau.to(buf.dtype)

    # ------------------------------------------------------------------ logging / getters
    def last_torque(self) -> Tensor:
        """tau_r of the last physics substep [N, J] (N m), order self.joint_names."""
        return self.tau

    def step_stats(self) -> Tuple[Tensor, Tensor]:
        """Mean |tau_r| and fraction of active substeps over the last control step, [N, J] each."""
        return self.step_abs_tau, self.step_active

    def log_values(self) -> Dict[str, Tensor]:
        out = {"joint_reflex/delay_ms": self.delay_ms.float().mean().reshape(1)}
        a, f = self.step_abs_tau.mean(0), self.step_active.mean(0)
        for i, j in enumerate(self.joint_names):
            out[f"joint_reflex/abs_tau_{j}"] = a[i].reshape(1)
            out[f"joint_reflex/active_{j}"] = f[i].reshape(1)
        return out

    def _histogram(self, x: Tensor, away: Tensor):
        c = self.config
        bw = c.hist_max / c.hist_bins
        b = (x / bw).long().clamp(0, c.hist_bins - 1)                  # [N, J]
        off = torch.arange(self.J, device=self.device) * c.hist_bins
        flat = (b + off).reshape(-1)
        self.hist.view(-1).scatter_add_(0, flat, torch.ones_like(flat))
        sel = flat[away.reshape(-1)]
        self.hist_away.view(-1).scatter_add_(0, sel, torch.ones_like(sel))

    def calibration_quantiles(self, qs=(0.5, 0.9, 0.99, 0.999)) -> dict:
        """|e_v| quantiles per joint (rad/s, bin upper edges) over all substeps and over 'away' substeps."""
        if self.hist is None:
            raise RuntimeError("calibrate=False")
        bw = self.config.hist_max / self.config.hist_bins
        res = {}
        for name, h in (("all", self.hist), ("away", self.hist_away)):
            cdf = h.cumsum(-1).double()
            tot = cdf[:, -1:].clamp_min(1)
            res[name] = {j: {str(q): float((torch.searchsorted(cdf[i] / tot[i], torch.tensor([q], dtype=torch.double,
                                        device=cdf.device)).clamp(max=h.shape[1] - 1).item() + 1) * bw)
                             for q in qs} for i, j in enumerate(self.joint_names)}
            res[name + "_samples"] = int(h[0].sum().item())
        return res

    def save_state(self) -> dict:
        return {k: getattr(self, k).clone() for k in ("history", "delay_steps", "q_ref", "qd_ref", "u", "fresh", "tau")} \
            | {"head": self.head}

    def load_state(self, state: dict):
        self.head = state["head"]
        for k, v in state.items():
            if k != "head":
                getattr(self, k).copy_(v)


def build_joint_reflex(config: Optional[JointReflexConfig], env) -> Optional[JointReflex]:
    """Env helper: None unless enabled. HC_JOINT_REFLEX_CALIBRATE=<path.json> (evaluation) builds a sensor-only block
    (gain 0, histograms) on any VIC run, also when the recipe has none, and writes |e_v| quantiles at exit."""
    calib = os.environ.get("HC_JOINT_REFLEX_CALIBRATE")
    if calib:
        base = config if (config is not None and config.enabled) else JointReflexConfig()
        config = JointReflexConfig(**{**base.__dict__, "enabled": True, "gain_scale": 0.0, "calibrate": True})
    if config is None or not config.enabled:
        return None
    rc = env.robot_config
    info = rc.control.control_info
    sim = env.simulator
    block = JointReflex(config, rc.kinematic_info.dof_names, {j: info[j].effort_limit for j in config.joints},
                        physics_hz=sim.config.sim.fps, control_dt=env.dt, num_envs=env.num_envs, device=env.device)
    if calib:
        def _dump():
            try:
                with open(calib, "w") as f:
                    json.dump(dict(block.calibration_quantiles(), joints=list(block.joint_names),
                                   physics_hz=float(sim.config.sim.fps)), f, indent=1)
            except Exception as exc:              # noqa: BLE001 - exit hook must not raise
                print(f"[joint_reflex] calibration dump failed: {exc}", flush=True)
        block._dump_calibration = _dump
        atexit.register(_dump)
    return block
