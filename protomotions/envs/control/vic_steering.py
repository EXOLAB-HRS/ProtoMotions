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
"""
from dataclasses import dataclass, field
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

# Human joint impedance ranges (k N m/rad, b N m s/rad) for swing / stance / standing, model 63.3 kg.
# Hip and knee in stance and standing: relaxed-static values (no walking stance measurement).
# Ankle swing: assumption (low end of the walking range).
HUMAN_RANGES = {
    "Hip_y": {"k": ((34, 66), (50, 318), (50, 318)), "b": ((1.9, 4.6), (2, 21), (2, 21))},
    "Knee_y": {"k": ((0.5, 3.5), (15, 75), (15, 75)), "b": ((0.02, 0.14), (1, 2.2), (1, 2.2))},
    "Ankle_y": {"k": ((22, 100), (22, 406), (290, 340)), "b": ((0.6, 1.5), (0.6, 1.5), (0.6, 1.5))},
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
        for i, j in enumerate(c.vic_joints):
            key = j[2:]
            if key in HUMAN_RANGES:
                for s in range(3):
                    lo, hi = HUMAN_RANGES[key]["k"][s]
                    blo, bhi = HUMAN_RANGES[key]["b"][s]
                    lk_h[i, s] = 0.5 * (math.log(lo) + math.log(hi))
                    lb_h[i, s] = 0.5 * (math.log(blo) + math.log(bhi))
                    half[i, s] = 0.5 * (math.log(hi) - math.log(lo)) + math.log(c.band_widen)
            else:
                lk_h[i] = math.log(float(k_old[i]))
                lb_h[i] = math.log(float(b_old[i]))
                half[i] = math.log(c.frontal_band)
        self._lk_h, self._lb_h, self._half = lk_h, lb_h, half
        n = len(self._tar_speed)
        self._lk = torch.log(k_old).expand(n, J).clone()
        self._lb = torch.log(b_old).expand(n, J).clone()
        self._fresh = torch.ones(n, dtype=torch.bool, device=dev)
        self.alpha = float(c.alpha_init)
        self.beta = float(c.beta_init)
        self._steps = 0
        self._fall_base = None
        self._fall_sum = 0.0
        self._fall_ema = None
        self._held = 0
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
        self._sim_ready = False

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
        self._sim_ready = True

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

    # ------------------------------------------------------------------ schedule
    def _advance_schedule(self):
        c = self.config
        self._steps += 1
        fell = self.env.terminate_buf.float().mean().item()
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
                   mean_kp_swing_stance_stand=mk, joints=list(c.vic_joints))
        print("[vic] " + json.dumps(rec), flush=True)
        if self._state_file:
            try:
                with open(self._state_file, "w") as f:
                    json.dump(rec, f)
            except OSError:
                pass

    # ------------------------------------------------------------------ main hook
    def reset(self, env_ids):
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
        delta = c.delta_max * torch.tanh(extra[:, :J])
        tau_res = c.tau_res_scale * self._tau_max * torch.tanh(extra[:, J:2 * J])
        # Leg state: 0 swing, 1 stance, 2 standing (both feet down, slow command and body).
        stance = self._stance[:, self._leg]                                   # [N, J]
        root_v = sim.get_robot_state().rigid_body_vel[:, 0, :2].norm(dim=-1)
        standing = self._stance.all(-1) & (self._tar_speed < c.stand_speed) & (root_v < c.stand_root_speed)
        state = torch.where(stance, torch.ones_like(stance, dtype=torch.long),
                            torch.zeros_like(stance, dtype=torch.long))
        state = torch.where(standing[:, None], torch.full_like(state, 2), state)
        jj = torch.arange(J, device=state.device).expand_as(state)
        lk_h = self._lk_h[jj, state]
        lb_h = self._lb_h[jj, state]
        a = self.alpha
        lk_t = (1 - a) * torch.log(self._k_old) + a * lk_h
        lb_t = (1 - a) * torch.log(self._b_old) + a * lb_h
        s = 1.0 - math.exp(-self.env.dt / c.gain_smooth_s)
        self._lk = torch.where(self._fresh[:, None], lk_t, self._lk + s * (lk_t - self._lk))
        self._lb = torch.where(self._fresh[:, None], lb_t, self._lb + s * (lb_t - self._lb))
        self._fresh[:] = False
        kp = torch.exp(self._lk + delta)
        kd = torch.exp(self._lb + delta)
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
        self._cocon = torch.relu(delta).mean(-1)
        tau = kp * err - kd * qd + self.beta * prior + sim.vic_ff_res[:, self._idx]
        self._effort = ((tau / self._tau_max) ** 2).clamp(max=4.0).mean(-1)
        for st in range(3):
            m = state == st
            cnt = m.float().sum(0).clamp_min(1)
            self._mean_k[st] = torch.where(m.any(0), ((self._lk + delta) * m).sum(0) / cnt, self._mean_k[st])
        self._log()

    def populate_context(self, ctx):
        super().populate_context(ctx)
        ctx.steering.vic_band_cost = self._band
        ctx.steering.vic_cocontraction = self._cocon
        ctx.steering.vic_effort = self._effort
