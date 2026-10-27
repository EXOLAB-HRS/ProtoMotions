"""Variable joint impedance for the teacher (vic_tc_v1, candidate, 2026-10-22).

The policy keeps its 59 PD targets and gets 2 * J extra outputs for J selected joints:
a stiffness residual delta and a feed-forward torque residual tau_res. Per joint,

    k_p = exp(lk + delta),  k_d = exp(lb + delta),  delta = delta_max * tanh(a)
    tau = tau_ff + k_p * (q_ref - q) - k_d * qdot          (PhysX implicit drive + feed-forward)
    tau_ff = beta * tau_prior + lowpass(tau_res),  tau_res = tau_res_scale * tau_max * tanh(b)
    tau_prior = (k_old - k_p) * (q_ref - q) - (b_old - k_d) * qdot   (control step, held)

lk, lb follow a reference that moves in log space from the current gains (alpha = 0) to a human
reference (alpha = 1) chosen by each leg's state (swing / stance / both feet standing), smoothed
over gain_smooth_s. Human references are geometric centres of literature ranges
(html/HumanModel_v3_1_261022_joint_impedance_vs_human_ghlee.html); joints without a human value
(frontal) keep the current gain as reference. tau_prior lets the slow part of the torque match the
parent (r25) gains while the drive itself is already soft; beta goes 1 -> 0 so the policy takes the
torque over. alpha and beta advance on a step schedule and pause while the fall rate is above the
baseline measured at alpha = 0.

State is per leg from the cadence stance rule (foot body heights), not from a gait clock, so it also
applies to standing, stopping and turning.

Candidate options (2026-10-26, off by default): delta_band (delta_max per joint and leg state from the human range)
and delta_lowpass_s (lowpass on delta) or delta_tau_up_s / delta_tau_down_s (activation-like asymmetric lowpass);
recipe vic_TC_r11.

vic_tc_v2 (2026-10-28, user-approved version: stiffness / damping should not jitter like a trajectory but take clearly
separated, physiologically reasonable levels for swing / stance / standing / disturbance response). Per joint (hip,
knee, ankle, frontal hip, subtalar), the existing residual becomes a co-activation level inside the human range:

    u_j = tanh(a_j) in [-1, 1],  delta_j = h_j(state) * u_j,  h_j = half log-width of the human range of state

with delta_band_margin = 1 (u = -1 / 0 / +1 gives the low end / geometric centre / high end of the range, never
outside it; v1 allowed x1.5 beyond), an activation-like filter (delta_tau_up_s / delta_tau_down_s, 50 / 150 ms in the
v2 recipes; v1 15 / 50 ms) and, with coact_unit_cost, the cocontraction cost on the level, mean_j relu(u_j), so it means
the same for every joint. The filtered level u_j = delta_j / h_j is written to the gain log as "u".
"""
from dataclasses import dataclass, field
import atexit
import json
import math
import os

import torch
from torch import Tensor

from protomotions.envs.control.cadence_gait_prior import (
    CadenceGaitPercentSteering, CadenceGaitPercentSteeringConfig, steering_path)

VIC_BAND_COST = steering_path("vic_band_cost")
VIC_COCONTRACTION = steering_path("vic_cocontraction")
VIC_EFFORT = steering_path("vic_effort")
# vic_TC_r2 (2026-10-23): foot impact above the human step peak, and a metabolic proxy of the sagittal leg joints.
VIC_IMPACT = steering_path("vic_impact")
VIC_METAB = steering_path("vic_metab")

# Human joint impedance ranges (k N m/rad, b N m s/rad) for swing / stance / standing, model 63.3 kg.
# Hip and knee in stance and standing: relaxed-static values (no walking stance measurement).
# Ankle swing: assumption (low end of the walking range).
HUMAN_RANGES = {
    "Hip_y": {"k": ((34, 66), (50, 318), (50, 318)), "b": ((1.9, 4.6), (2, 21), (2, 21))},
    "Knee_y": {"k": ((0.5, 3.5), (15, 75), (15, 75)), "b": ((0.02, 0.14), (1, 2.2), (1, 2.2))},
    "Ankle_y": {"k": ((22, 100), (22, 406), (290, 340)), "b": ((0.6, 1.5), (0.6, 1.5), (0.6, 1.5))},
}
# Frontal joints (candidate, 2026-10-26, used only with frontal_human_ref): scale-level ranges, not phase references
# (share/HMV_v1_teacher_261026/human_impedance_ref_3state.csv, frontal rows).
# Hip_x stance: WBDS frontal hip quasi-stiffness (moment / angle slope, 0-50 % cycle, 753 cycles, p10-p90 167-679, median
#   335 N m/rad at 63.3 kg); standing: Winter 1998 quiet-stance M/L stiffness 627-1137 N m/rad shared by two hips;
#   swing: assumption = sagittal hip swing. Damping: assumption = sagittal hip.
# Subtalar_x (ankle inversion-eversion): relaxed 8.8 +- 2.2 N m/rad, 0.18 +- 0.09 N m s/rad (Ho, Lee, Hogan 2016);
#   active stance: relaxed + 1.1 N m/rad per % MVC (Lee, Krebs, Hogan 2014) at 10-50 % MVC, inside the WBDS frontal ankle
#   quasi-stiffness p10-p90 10-146. Damping above relaxed: assumption.
HUMAN_RANGES_FRONTAL = {
    "Hip_x": {"k": ((34, 66), (100, 700), (300, 600)), "b": ((1.9, 4.6), (2, 21), (2, 21))},
    "Subtalar_x": {"k": ((5, 15), (10, 70), (10, 30)), "b": ((0.1, 0.3), (0.2, 1.5), (0.2, 0.6))},
}


def vic_cost(value: Tensor) -> Tensor:
    """Penalty reward: -value (value >= 0)."""
    return -value


@dataclass
class VicSteeringConfig(CadenceGaitPercentSteeringConfig):
    _target_: str = "protomotions.envs.control.vic_steering.VicSteering"
    vic_joints: tuple = ("L_Hip_y", "L_Knee_y", "L_Ankle_y", "R_Hip_y", "R_Knee_y", "R_Ankle_y",
                         "L_Hip_x", "R_Hip_x", "L_Subtalar_x", "R_Subtalar_x")
    delta_max: float = 1.1
    tau_res_scale: float = 0.33          # fraction of the joint strength cap
    tau_ff_lowpass_s: float = 0.03       # activation-like lowpass on tau_res (per physics substep)
    gain_smooth_s: float = 0.1
    band_widen: float = 1.5              # tolerance: human range widened by this factor each side
    frontal_band: float = 2.0            # tolerance for joints whose reference is the current gain
    stand_speed: float = 0.1
    stand_root_speed: float = 0.25
    # Schedule in control steps (one env.step = one control step for all envs).
    baseline_steps: int = 600
    alpha_steps: int = 9000
    beta_steps: int = 9000
    beta_decay: bool = True
    alpha_init: float = 0.0
    beta_init: float = 1.0
    schedule: bool = True
    fall_ratio_tol: float = 1.5
    fall_abs_tol: float = 0.0005
    log_every: int = 200
    # vic_TC_r2 cost inputs (rewards only exist when a recipe adds them). Human walking step peak of the vertical
    # GRF is 1.1-1.3 body weights; metabolic proxy P+/0.25 + |P-|/1.20 (muscle efficiencies, Margaria 1968) of
    # Hip_y, Knee_y, Ankle_y, in units of m*g*(1 m/s).
    body_mass: float = 63.3
    grf_ref_bw: float = 1.3
    grf_bodies: tuple = (("L_Ankle", "L_Toe"), ("R_Ankle", "R_Toe"))
    # k_d = exp(lb + damp_delta_gain * delta): 1 scales damping with stiffness (zeta grows as sqrt(k)); 0.5 keeps the
    # damping ratio zeta of the reference while the policy changes k (k_d ~ sqrt(k_p)), vic_TC_r4 (2026-10-23).
    damp_delta_gain: float = 1.0
    fresh_stance: bool = False           # contact from foot heights on the first step after a reset (r2 trains with it)
    # Candidate change (2026-10-26, HMV_v1 meeting follow-up Q2; defaults keep r1-r10 behaviour):
    # delta_band: delta_max per joint and leg state = half log-width of the human range + ln(delta_band_margin), so
    #   every state can reach its whole range (ankle stance needs 1.46 > 1.1); joints without a human value keep
    #   delta_max. The bound follows the smoothed state weights, so it changes as smoothly as the reference.
    # delta_lowpass_s: first-order lowpass of delta per control step (muscle activation ~30-50 ms); 0 = off.
    delta_band: bool = False
    frontal_human_ref: bool = False      # Hip_x / Subtalar_x follow HUMAN_RANGES_FRONTAL instead of the current gain
    delta_band_margin: float = 1.5
    delta_lowpass_s: float = 0.0
    # Activation-like asymmetric lowpass (2026-10-26, overrides delta_lowpass_s when both > 0): delta read as the
    # co-contraction level; rising (stiffening) follows tau_up, falling (relaxing) tau_down, like muscle activation
    # vs deactivation (~10-20 ms vs ~30-60 ms, Zajac 1989; Thelen 2003). Gains change once per control step.
    delta_tau_up_s: float = 0.0
    delta_tau_down_s: float = 0.0
    coact_unit_cost: bool = False        # vic_tc_v2: cocontraction cost on the level u = delta / h (not on delta)
    # Exoskeleton hip assist / resist randomisation during training (2026-10-28, candidate change, off by default; user:
    # implement now, run decided later). Same gait-phase profile as the W4 evaluation (scripts/hc_mc_cb_rollout.py
    # HC_EXO_PHASE): per leg gait % p, extension bump +A_ext sin(pi p / 0.3) on 0-30 % and flexion bump
    # -A_flex sin(pi (p - 0.45) / 0.3) on 45-75 % [N m] on Hip_y, added outside the muscle caps (an exoskeleton motor),
    # only while the leg has a stride estimate and the command is faster than exo_min_speed. Per env, a new segment every
    # exo_interval_range s: off with exo_off_prob, resist with exo_resist_prob, else assist; A_ext, A_flex ~
    # U(exo_amp_range) with the segment's sign. The policy does not observe the exo torque (as with a real device).
    exo_rand: bool = False
    exo_amp_range: tuple = (5.0, 20.0)
    exo_off_prob: float = 0.3
    exo_resist_prob: float = 0.3
    exo_interval_range: tuple = (4.0, 8.0)
    exo_min_speed: float = 0.3


class VicSteering(CadenceGaitPercentSteering):
    def __init__(self, config, env):
        super().__init__(config, env)
        c = config
        sim = env.simulator
        names = list(env.robot_config.kinematic_info.dof_names)
        dev = self._tar_speed.device
        self._idx = torch.tensor([names.index(j) for j in c.vic_joints], device=dev)
        self._leg = torch.tensor([0 if j.startswith("L_") else 1 for j in c.vic_joints], device=dev)
        J = len(c.vic_joints)
        self.extra_action_dim = 2 * J
        info = env.robot_config.control.control_info
        k_old = torch.tensor([info[j].stiffness for j in c.vic_joints], device=dev, dtype=torch.float)
        b_old = torch.tensor([info[j].damping for j in c.vic_joints], device=dev, dtype=torch.float)
        tau_max = torch.tensor([info[j].effort_limit for j in c.vic_joints], device=dev, dtype=torch.float)
        self._k_old, self._b_old, self._tau_max = k_old, b_old, tau_max
        lk_h = torch.zeros(J, 3, device=dev)
        lb_h = torch.zeros(J, 3, device=dev)
        half = torch.zeros(J, 3, device=dev)       # log half-width of the tolerated band
        ranges = dict(HUMAN_RANGES, **(HUMAN_RANGES_FRONTAL if c.frontal_human_ref else {}))
        for i, j in enumerate(c.vic_joints):
            key = j[2:]
            if key in ranges:
                for s in range(3):
                    lo, hi = ranges[key]["k"][s]
                    blo, bhi = ranges[key]["b"][s]
                    lk_h[i, s] = 0.5 * (math.log(lo) + math.log(hi))
                    lb_h[i, s] = 0.5 * (math.log(blo) + math.log(bhi))
                    half[i, s] = 0.5 * (math.log(hi) - math.log(lo)) + math.log(c.band_widen)
            else:
                lk_h[i] = math.log(float(k_old[i]))
                lb_h[i] = math.log(float(b_old[i]))
                half[i] = math.log(c.frontal_band)
        self._lk_h, self._lb_h, self._half = lk_h, lb_h, half
        dmax = torch.full((J, 3), float(c.delta_max), device=dev)
        if c.delta_band:
            for i, j in enumerate(c.vic_joints):
                key = j[2:]
                if key in ranges:
                    for s in range(3):
                        lo, hi = ranges[key]["k"][s]
                        dmax[i, s] = 0.5 * (math.log(hi) - math.log(lo)) + math.log(c.delta_band_margin)
        self._dmax_h = dmax
        n = len(self._tar_speed)
        self._lk = torch.log(k_old).expand(n, J).clone()
        self._lb = torch.log(b_old).expand(n, J).clone()
        self._fresh = torch.ones(n, dtype=torch.bool, device=dev)
        self._dmax = dmax[:, 0].expand(n, J).clone()          # smoothed per-env bound (delta_band)
        self._delta = torch.zeros(n, J, device=dev)            # lowpassed delta (delta_lowpass_s)
        self.alpha = float(c.alpha_init)
        self.beta = float(c.beta_init)
        self._steps = 0
        self._fall_base = None
        self._fall_sum = 0.0
        self._fall_ema = None
        self._held = 0
        self._fall_acc = 0
        self._band = torch.zeros(n, device=dev)
        self._cocon = torch.zeros(n, device=dev)
        self._effort = torch.zeros(n, device=dev)
        self._mean_k = torch.zeros(3, J, device=dev)
        self._state_file = os.environ.get("HC_VIC_STATE_FILE")
        # HC_VIC_FIXED="alpha,beta" holds both values (evaluation of a trained checkpoint, smoke tests).
        fixed = os.environ.get("HC_VIC_FIXED")
        self._fixed = bool(fixed)
        if fixed:
            self.alpha, self.beta = (float(x) for x in fixed.split(","))
        # HC_VIC_FRESH_STANCE=1 (evaluation, 2026-10-23): on the first control step after a reset the cadence
        # contact flags are still False (both legs "swing", knee k ~ 2 N m/rad); take them from foot heights instead.
        self._fresh_stance = c.fresh_stance or os.environ.get("HC_VIC_FRESH_STANCE", "0") == "1"
        self._sag = torch.tensor([i for i, j in enumerate(c.vic_joints) if j[2:] in HUMAN_RANGES], device=dev)
        self._bw = c.body_mass * 9.81
        self._impact = torch.zeros(n, device=dev)
        self._metab = torch.zeros(n, device=dev)
        self._sim_ready = False
        if c.exo_rand:
            self._exo_hips = torch.tensor([names.index(j) for j in ("L_Hip_y", "R_Hip_y")], device=dev)
            self._exo_amp = torch.zeros(n, 2, device=dev)               # (A_ext, A_flex) with the segment's sign
            self._exo_left = torch.zeros(n, device=dev)                 # control steps until the next segment
            self._exo_tau = torch.zeros(n, 2, device=dev)               # applied torque (L_Hip_y, R_Hip_y)
        # HC_VIC_GAIN_LOG=<path.npz> (evaluation, 2026-10-27, off by default): k_p, k_d, delta and leg state of the VIC
        # joints at every control step, written at exit (jitter analysis of the stiffness output).
        self._gain_log_path = os.environ.get("HC_VIC_GAIN_LOG")
        self._gain_log = []
        self._u_log = []                                       # vic_tc_v2: filtered co-activation level u [T, N, J]
        if self._gain_log_path:
            atexit.register(self._dump_gain_log)

    # ------------------------------------------------------------------ simulator buffers
    def _setup_sim(self):
        sim = self.env.simulator
        n = len(self._tar_speed)
        p, d = sim._common_p_gains, sim._common_d_gains
        if p.dim() == 1:
            sim._common_p_gains = p.unsqueeze(0).expand(n, -1).clone()
            sim._common_d_gains = d.unsqueeze(0).expand(n, -1).clone()
        ndof = sim._common_p_gains.shape[1]
        z = torch.zeros(n, ndof, device=p.device)
        sim.vic_ff_prior, sim.vic_ff_res_target, sim.vic_ff_res = z.clone(), z.clone(), z.clone()
        dt = 1.0 / sim.config.sim.fps
        sim.vic_ff_coeff = torch.tensor(1.0 - math.exp(-dt / max(self.config.tau_ff_lowpass_s, 1e-6)),
                                        device=p.device)
        robot = sim._robot
        self._to_sim = sim.data_conversion.dof_convert_to_sim
        self._host_k = torch.empty((n, ndof), dtype=torch.float, pin_memory=True)
        self._host_d = torch.empty((n, ndof), dtype=torch.float, pin_memory=True)
        self._host_idx = robot._ALL_INDICES.cpu()
        if self.config.exo_rand:
            self._exo_sim = torch.zeros(n, ndof, device=p.device)   # simulator dof order, updated in place
            apply_ = sim._apply_simulator_torques

            def _apply_with_exo(torques, *a, **k):
                return apply_(torques + self._exo_sim, *a, **k)
            sim._apply_simulator_torques = _apply_with_exo
        self._sim_ready = True

    def _exo_update(self):
        """New exo segments where due (or after a reset), then this control step's hip torque from each leg's gait %."""
        c = self.config
        env = self.env
        n, dev = len(self._tar_speed), self._tar_speed.device
        self._exo_left -= 1
        new = (self._exo_left <= 0) | (env.progress_buf <= 1)
        if new.any():
            r = torch.rand(n, device=dev)
            sign = torch.where(r < c.exo_off_prob, torch.zeros_like(r),
                               torch.where(r < c.exo_off_prob + c.exo_resist_prob, -torch.ones_like(r), torch.ones_like(r)))
            lo, hi = c.exo_amp_range
            amp = (lo + (hi - lo) * torch.rand(n, 2, device=dev)) * sign[:, None]
            t0, t1 = c.exo_interval_range
            left = (t0 + (t1 - t0) * torch.rand(n, device=dev)) / env.dt
            self._exo_amp = torch.where(new[:, None], amp, self._exo_amp)
            self._exo_left = torch.where(new, left, self._exo_left)
        now = env.progress_buf.float()[:, None]
        pct = (now - self._leg_onset) / self._leg_stride.clamp_min(1.)                 # [n, 2] (L, R)
        valid = (self._leg_strides >= 2) & (pct <= 1.2) & (self._tar_speed[:, None] > c.exo_min_speed)
        ext = torch.where((pct >= 0) & (pct < 0.3), torch.sin(math.pi * pct / 0.3), torch.zeros_like(pct))
        flx = torch.where((pct >= 0.45) & (pct < 0.75), torch.sin(math.pi * (pct - 0.45) / 0.3), torch.zeros_like(pct))
        self._exo_tau = (self._exo_amp[:, :1] * ext - self._exo_amp[:, 1:] * flx) * valid
        vec = torch.zeros(n, len(self.env.robot_config.kinematic_info.dof_names), device=dev)
        vec[:, self._exo_hips] = self._exo_tau
        self._exo_sim.copy_(vec[:, self._to_sim])

    def _write_gains(self):
        sim = self.env.simulator
        robot = sim._robot
        k_sim = sim._common_p_gains[:, self._to_sim]
        d_sim = sim._common_d_gains[:, self._to_sim]
        robot._data.joint_stiffness.copy_(k_sim)
        robot._data.joint_damping.copy_(d_sim)
        self._host_k.copy_(k_sim)
        self._host_d.copy_(d_sim)
        robot.root_physx_view.set_dof_stiffnesses(self._host_k, indices=self._host_idx)
        robot.root_physx_view.set_dof_dampings(self._host_d, indices=self._host_idx)

    def _dump_gain_log(self):
        if not self._gain_log:
            return
        import numpy as np
        a = torch.stack(self._gain_log, 0).float().numpy()          # [T, 4, N, J]
        extra = {"u": torch.stack(self._u_log[:len(a)], 0).float().numpy()} if self._u_log else {}
        np.savez_compressed(self._gain_log_path, kp=a[:, 0], kd=a[:, 1], delta=a[:, 2], state=a[:, 3].astype("int8"),
                            joints=np.array(self.config.vic_joints), dt=float(self.env.dt), **extra)

    # ------------------------------------------------------------------ schedule
    def _advance_schedule(self):
        c = self.config
        self._steps += 1
        # Falls per env-step, counted when an env is reset after a fall termination (same quantity as the
        # training log's termination/fallen_mean). Reading terminate_buf before physics gave ~20x that value.
        fell = self._fall_acc / len(self._tar_speed)
        self._fall_acc = 0
        if self._steps <= c.baseline_steps:
            self._fall_sum += fell
            if self._steps == c.baseline_steps:
                self._fall_base = self._fall_sum / c.baseline_steps
                self._fall_ema = self._fall_base
            return
        if self._fall_ema is None:
            self._fall_ema = fell
        self._fall_ema += (fell - self._fall_ema) / 300.0
        if not c.schedule or self._fixed or self._fall_base is None:
            return
        if self._fall_ema > c.fall_ratio_tol * self._fall_base + c.fall_abs_tol:
            self._held += 1
            return
        if self.alpha < 1.0:
            self.alpha = min(1.0, self.alpha + 1.0 / c.alpha_steps)
        elif c.beta_decay and self.beta > 0.0:
            self.beta = max(0.0, self.beta - 1.0 / c.beta_steps)

    def _log(self):
        c = self.config
        if self._steps % c.log_every:
            return
        mk = self._mean_k.exp().cpu().numpy().round(2).tolist()
        rec = dict(step=self._steps, alpha=round(self.alpha, 4), beta=round(self.beta, 4),
                   fall_rate=None if self._fall_ema is None else round(self._fall_ema, 5),
                   fall_base=None if self._fall_base is None else round(self._fall_base, 5),
                   held_steps=self._held, band_cost=round(self._band.mean().item(), 4),
                   cocontraction=round(self._cocon.mean().item(), 4), effort=round(self._effort.mean().item(), 4),
                   impact=round(self._impact.mean().item(), 4), metab=round(self._metab.mean().item(), 4),
                   mean_kp_swing_stance_stand=mk, joints=list(c.vic_joints))
        if c.exo_rand:
            rec.update(exo_abs_tau=round(self._exo_tau.abs().mean().item(), 3),
                       exo_assist_frac=round((self._exo_amp[:, 0] > 0).float().mean().item(), 3),
                       exo_resist_frac=round((self._exo_amp[:, 0] < 0).float().mean().item(), 3))
        print("[vic] " + json.dumps(rec), flush=True)
        if self._state_file:
            try:
                with open(self._state_file, "w") as f:
                    json.dump(rec, f)
            except OSError:
                pass

    # ------------------------------------------------------------------ main hook
    def reset(self, env_ids):
        if len(env_ids) and hasattr(self.env, "terminate_buf"):
            # Called before the env clears terminate_buf; skip envs that never ran (initial reset).
            fell = self.env.terminate_buf[env_ids] & (self.env.progress_buf[env_ids] > 2)
            self._fall_acc = getattr(self, "_fall_acc", 0) + int(fell.sum().item())
        super().reset(env_ids)
        if len(env_ids):
            self._fresh[env_ids] = True

    def pre_physics_extra_action(self, extra: Tensor, q_ref: Tensor):
        """Called by the env each control step before physics, with the raw extra outputs and the
        processed PD targets (common dof order)."""
        c = self.config
        if not self._sim_ready:
            self._setup_sim()
        self._advance_schedule()
        sim = self.env.simulator
        J = len(c.vic_joints)
        unit = torch.tanh(extra[:, :J])
        tau_res = c.tau_res_scale * self._tau_max * torch.tanh(extra[:, J:2 * J])
        # Leg state: 0 swing, 1 stance, 2 standing (both feet down, slow command and body).
        contact = self._stance
        if self._fresh_stance and self._fresh.any():
            height = sim.get_robot_state().rigid_body_pos[..., 2]
            measured = torch.stack([(height[:, idx] < self._stance_heights).any(-1) for idx in self._feet], dim=-1)
            contact = torch.where(self._fresh[:, None], measured, contact)
        stance = contact[:, self._leg]                                        # [N, J]
        root_v = sim.get_robot_state().rigid_body_vel[:, 0, :2].norm(dim=-1)
        standing = contact.all(-1) & (self._tar_speed < c.stand_speed) & (root_v < c.stand_root_speed)
        state = torch.where(stance, torch.ones_like(stance, dtype=torch.long),
                            torch.zeros_like(stance, dtype=torch.long))
        state = torch.where(standing[:, None], torch.full_like(state, 2), state)
        jj = torch.arange(J, device=state.device).expand_as(state)
        if c.exo_rand:
            self._exo_update()
        lk_h = self._lk_h[jj, state]
        lb_h = self._lb_h[jj, state]
        a = self.alpha
        lk_t = (1 - a) * torch.log(self._k_old) + a * lk_h
        lb_t = (1 - a) * torch.log(self._b_old) + a * lb_h
        s = 1.0 - math.exp(-self.env.dt / c.gain_smooth_s)
        self._lk = torch.where(self._fresh[:, None], lk_t, self._lk + s * (lk_t - self._lk))
        self._lb = torch.where(self._fresh[:, None], lb_t, self._lb + s * (lb_t - self._lb))
        fresh = self._fresh.clone()
        self._fresh[:] = False
        if c.delta_band:
            dm_t = self._dmax_h[jj, state]
            self._dmax = torch.where(fresh[:, None], dm_t, self._dmax + s * (dm_t - self._dmax))
            delta = self._dmax * unit
        else:
            delta = c.delta_max * unit
        if c.delta_tau_up_s > 0 and c.delta_tau_down_s > 0:
            up = 1.0 - math.exp(-self.env.dt / c.delta_tau_up_s)
            down = 1.0 - math.exp(-self.env.dt / c.delta_tau_down_s)
            step = torch.where(delta > self._delta, up, down) * (delta - self._delta)
            self._delta = torch.where(fresh[:, None], delta, self._delta + step)
            delta = self._delta
        elif c.delta_lowpass_s > 0:
            sd = 1.0 - math.exp(-self.env.dt / c.delta_lowpass_s)
            self._delta = torch.where(fresh[:, None], delta, self._delta + sd * (delta - self._delta))
            delta = self._delta
        kp = torch.exp(self._lk + delta)
        kd = torch.exp(self._lb + c.damp_delta_gain * delta)
        if self._gain_log_path:
            self._gain_log.append(torch.stack([kp, kd, delta, state.float()], 0).half().cpu())
            if c.coact_unit_cost:
                self._u_log.append((delta / self._dmax.clamp_min(1e-6)).half().cpu())
            if len(self._gain_log) % 250 == 0:      # the simulator may exit without running atexit hooks
                self._dump_gain_log()
        sim._common_p_gains[:, self._idx] = kp
        sim._common_d_gains[:, self._idx] = kd
        self._write_gains()
        dof = sim._get_simulator_dof_state().convert_to_common(sim.data_conversion)
        q, qd = dof.dof_pos[:, self._idx], dof.dof_vel[:, self._idx]
        err = q_ref[:, self._idx] - q
        prior = (self._k_old - kp) * err - (self._b_old - kd) * qd
        sim.vic_ff_prior[:, self._idx] = self.beta * prior
        sim.vic_ff_res_target[:, self._idx] = tau_res
        reset_now = self.env.progress_buf <= 1
        if reset_now.any():
            sim.vic_ff_res[reset_now] = 0.0
        # Reward inputs.
        dev_ = (self._lk + delta) - lk_t
        self._band = (torch.relu(dev_.abs() - self._half[jj, state]) / math.log(2.0)).clamp(max=4.0).mean(-1)
        self._cocon = (torch.relu(delta / self._dmax.clamp_min(1e-6)) if c.coact_unit_cost else torch.relu(delta)).mean(-1)
        tau = kp * err - kd * qd + self.beta * prior + sim.vic_ff_res[:, self._idx]
        self._effort = ((tau / self._tau_max) ** 2).clamp(max=4.0).mean(-1)
        p = (tau * qd)[:, self._sag]
        self._metab = (p.clamp(min=0).sum(-1) / 0.25 + (-p).clamp(min=0).sum(-1) / 1.20) / self._bw
        for st in range(3):
            m = state == st
            cnt = m.float().sum(0).clamp_min(1)
            self._mean_k[st] = torch.where(m.any(0), ((self._lk + delta) * m).sum(0) / cnt, self._mean_k[st])
        self._log()

    def _update_impact(self):
        """Per leg, peak vertical contact force over the physics substeps of the last control step (body weights);
        cost = mean over legs of max(0, peak - grf_ref_bw), clamped at 3."""
        sm = getattr(self.env.simulator, "_contact_sensor_map", None)
        if not sm:
            return
        legs = [sum(sm[b].data.net_forces_w_history[:, :, 0, 2] for b in bodies).amax(-1) for bodies in self.config.grf_bodies]
        peak = torch.stack(legs, -1) / self._bw
        self._impact = torch.relu(peak - self.config.grf_ref_bw).clamp(max=3.0).mean(-1)

    def populate_context(self, ctx):
        super().populate_context(ctx)
        self._update_impact()
        ctx.steering.vic_impact = self._impact
        ctx.steering.vic_metab = self._metab
        ctx.steering.vic_band_cost = self._band
        ctx.steering.vic_cocontraction = self._cocon
        ctx.steering.vic_effort = self._effort
