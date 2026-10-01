"""Candidate cadence-commanded steering owned by the c-a7-cadence experiment.

Adds a cadence-ratio command rho* to ContinuousSteering. The target cadence is
f* = rho* * (f0_slope * v* + f0_intercept) in steps/s, with f0 the pack's natural
cadence at the commanded speed v* (scripts/cadence_labels.py). Below min_speed and
for stop commands rho* is 1, matching the reference labels of non-walking frames.
The measured cadence counts alternating foot-contact onsets of the simulated body.
The robot has no contact sensors (contact_bodies is None, so rigid_body_contacts is
all zero), so a foot is in contact while its Ankle or Toe body origin is below a
height threshold. On the forward25 pack this height rule reproduces the labelled
contacts' clip cadence within a 4.5% median error (thresholds 0.055 / 0.040 m).
In the v3.1 plant the Ankle origin stays above 0.065 m (260929 debug smoke), so Toe
onsets alone mark the steps there; alternation still gives one onset per step.
"""
from dataclasses import dataclass
import os
import torch
from protomotions.envs.control.continuous_steering import ContinuousSteering, ContinuousSteeringConfig

_DEBUG = os.environ.get("HC_CADENCE_DEBUG") == "1"


@dataclass
class CadenceSteeringConfig(ContinuousSteeringConfig):
    _target_: str = "protomotions.envs.control.cadence_steering.CadenceSteering"
    cadence_ratio_min: float = 0.9
    cadence_ratio_max: float = 1.1
    # Share of resamples that keep the natural gait (rho* = 1).
    cadence_natural_fraction: float = 0.3
    cadence_f0_slope: float = 0.558
    cadence_f0_intercept: float = 1.060
    cadence_min_speed: float = 0.5
    left_foot_bodies: tuple[str, ...] = ("L_Ankle", "L_Toe")
    right_foot_bodies: tuple[str, ...] = ("R_Ankle", "R_Toe")
    # Contact height of each foot body origin above flat ground, same order as the bodies.
    foot_contact_heights: tuple[float, ...] = (0.055, 0.040)


class CadenceSteering(ContinuousSteering):
    def __init__(self, config, env):
        super().__init__(config, env)
        if not 0 < config.cadence_ratio_min <= 1 <= config.cadence_ratio_max:
            raise ValueError("Cadence ratio range must contain 1")
        if not 0 <= config.cadence_natural_fraction <= 1:
            raise ValueError("cadence_natural_fraction must be in [0, 1]")
        names = list(env.robot_config.kinematic_info.body_names)
        device = self._tar_speed.device
        self._feet = [torch.tensor([names.index(b) for b in bodies], device=device)
                      for bodies in (config.left_foot_bodies, config.right_foot_bodies)]
        if len(config.foot_contact_heights) != len(config.left_foot_bodies):
            raise ValueError("One contact height per foot body")
        self._contact_heights = torch.tensor(config.foot_contact_heights, device=device)
        n = len(self._tar_speed)
        self._ratio = torch.ones(n, device=device)
        self._contact = torch.zeros(n, 2, dtype=torch.bool, device=device)
        self._last_side = torch.full((n,), -1, dtype=torch.long, device=device)
        self._last_strike = torch.zeros(n, device=device)
        self._intervals = torch.ones(n, 2, device=device)
        self._strikes = torch.zeros(n, dtype=torch.long, device=device)
        self._measured = torch.zeros(n, device=device)

    def reset(self, env_ids):
        super().reset(env_ids)
        if len(env_ids) == 0:
            return
        self._contact[env_ids] = False
        self._last_side[env_ids] = -1
        self._last_strike[env_ids] = self.env.progress_buf[env_ids].float()
        self._intervals[env_ids] = 1.
        self._strikes[env_ids] = 0
        self._measured[env_ids] = 0.

    def _resample_task(self, env_ids):
        super()._resample_task(env_ids)
        if len(env_ids) == 0:
            return
        c = self.config
        n, device = len(env_ids), env_ids.device
        ratio = c.cadence_ratio_min + torch.rand(n, device=device) * (c.cadence_ratio_max - c.cadence_ratio_min)
        natural = (torch.rand(n, device=device) < c.cadence_natural_fraction) | (self._goal_speed[env_ids] <= 0)
        self._ratio[env_ids] = torch.where(natural, torch.ones_like(ratio), ratio)

    def step(self):
        super().step()
        self.update_gait()

    def update_gait(self):
        """Foot contacts and measured cadence. Separate from step() so evaluations that
        freeze the command schedule (scripts/hc_eval_steering.py) still run it."""
        height = self.env.simulator.get_robot_state().rigid_body_pos[..., 2]
        now = self.env.progress_buf.float()
        contact = torch.stack([(height[:, idx] < self._contact_heights).any(-1) for idx in self._feet], dim=-1)
        onset = contact & ~self._contact
        self._contact[:] = contact
        for side in (0, 1):
            strike = onset[:, side] & (self._last_side != side)
            interval = now - self._last_strike
            counted = strike & (self._last_side >= 0)
            self._intervals[counted] = torch.stack(
                (self._intervals[counted, 1], interval[counted]), dim=-1)
            self._strikes += counted.long()
            self._last_strike[strike] = now[strike]
            self._last_side[strike] = side
        # A missing onset lowers the estimate as time passes, so not stepping cannot hold it.
        period = torch.maximum(self._intervals.mean(-1), now - self._last_strike).clamp_min(1.)
        self._measured[:] = 1. / (period * self.env.dt)
        if _DEBUG and int(self.env.progress_buf.max()) % 30 == 0:
            print(f"[cadence-debug] contact L/R {contact.float().mean(0).tolist()} "
                  f"onset {onset.float().mean().item():.3f} strikes {self._strikes.float().mean().item():.2f} "
                  f"measured p50 {self._measured.median().item():.2f} target p50 {self.target_ratio().median().item():.2f} "
                  f"foot z min L/R {[round(height[:, i].min().item(), 3) for i in self._feet[0]]} "
                  f"{[round(height[:, i].min().item(), 3) for i in self._feet[1]]} progress {int(self.env.progress_buf.max())}", flush=True)

    def target_ratio(self):
        walking = self._tar_speed >= self.config.cadence_min_speed
        return torch.where(walking, self._ratio, torch.ones_like(self._ratio))

    def populate_context(self, ctx):
        super().populate_context(ctx)
        c = self.config
        ratio = self.target_ratio()
        ctx.steering.tar_cadence_ratio = ratio
        ctx.steering.tar_cadence = ratio * (c.cadence_f0_slope * self._tar_speed + c.cadence_f0_intercept)
        ctx.steering.measured_cadence = self._measured
        ctx.steering.cadence_valid = (self._tar_speed >= c.cadence_min_speed) & (self._strikes >= 2)


@dataclass
class CadencePhaseSteeringConfig(CadenceSteeringConfig):
    """teacher_cadence_v1: cadence command given to the policy as a gait clock."""
    _target_: str = "protomotions.envs.control.cadence_steering.CadencePhaseSteering"
    # Middle share of each half cycle in which the clock's stance foot must be down.
    phase_swing_window: tuple[float, float] = (0.15, 0.85)
    # Stance (whole foot-down interval) per body, same order as the foot bodies. The strike
    # rule above (0.055 / 0.040 m) only catches the lowest instant of a step in the 240 Hz
    # plant; at these heights pc_240_4_r3-family walking shows ~59% stance per foot, 18%
    # double support, no flight and one onset per step (261001 e02 clock diagnosis).
    stance_heights: tuple[float, ...] = (0.078, 0.053)


class CadencePhaseSteering(CadenceSteering):
    """Gait clock phi advancing at pi * f* rad/s (one cycle = two steps).

    phi in [0, pi) is the left foot's stance half, [pi, 2 pi) the right foot's. The clock
    starts at the first stance onset after a reset or after it becomes active (walking
    command after a stop): phi = 0 at a left onset, pi at a right one. It is not re-aligned
    on command changes, so a policy that ignores it drifts out of phase and loses reward
    (re-aligning at every resample made ignoring it nearly free; 261001 e02 diagnosis).
    Below min_speed the clock is inactive.
    """

    def __init__(self, config, env):
        super().__init__(config, env)
        n, device = len(self._tar_speed), self._tar_speed.device
        if len(config.stance_heights) != len(config.left_foot_bodies):
            raise ValueError("One stance height per foot body")
        self._stance_heights = torch.tensor(config.stance_heights, device=device)
        self._stance = torch.zeros(n, 2, dtype=torch.bool, device=device)
        self._phase = torch.zeros(n, device=device)
        self._synced = torch.zeros(n, dtype=torch.bool, device=device)

    def reset(self, env_ids):
        super().reset(env_ids)
        if len(env_ids):
            self._synced[env_ids] = False
            self._stance[env_ids] = False

    def update_gait(self):
        super().update_gait()
        height = self.env.simulator.get_robot_state().rigid_body_pos[..., 2]
        stance = torch.stack([(height[:, idx] < self._stance_heights).any(-1) for idx in self._feet], dim=-1)
        onset = stance & ~self._stance
        self._stance[:] = stance
        active = self.clock_active()
        self._synced &= active
        start = ~self._synced & active & onset.any(-1)
        self._phase[start] = torch.where(onset[start, 0], 0., torch.pi)
        running = self._synced & active
        c = self.config
        rate = torch.pi * self.target_ratio() * (c.cadence_f0_slope * self._tar_speed + c.cadence_f0_intercept)
        self._phase[running] = torch.remainder(self._phase[running] + rate[running] * self.env.dt, 2 * torch.pi)
        self._synced |= start

    def clock_active(self):
        return self._tar_speed >= self.config.cadence_min_speed

    def populate_context(self, ctx):
        super().populate_context(ctx)
        active = self.clock_active() & self._synced
        ctx.steering.cadence_phase = torch.stack((self._phase.sin(), self._phase.cos()), -1) * active[:, None]
        half = torch.remainder(self._phase, torch.pi) / torch.pi
        lo, hi = self.config.phase_swing_window
        ctx.steering.phase_single_support = active & (half >= lo) & (half <= hi)
        ctx.steering.phase_stance_left = self._phase < torch.pi
        ctx.steering.foot_contact = self._stance
