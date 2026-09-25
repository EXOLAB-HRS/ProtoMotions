"""Candidate continuous-command control owned by b1-a7.

Uses the existing steering context and observations, so a7 weights stay compatible.
"""
from dataclasses import dataclass
import math
import torch
from protomotions.envs.control.steering_control import SteeringControl, SteeringControlConfig


@dataclass
class ContinuousSteeringConfig(SteeringControlConfig):
    _target_: str = "protomotions.envs.control.continuous_steering.ContinuousSteering"
    fixed_fraction: float = 0.3
    turn_fraction: float = 0.2
    acceleration_min: float = 0.2
    acceleration_max: float = 0.8
    yaw_rate: float = 0.35
    turn_angle_max: float = math.pi / 4
    full_heading_for_turn: bool = False
    independent_facing_fraction: float = 0.0
    # Only continuous groups sample stops; other targets use tar_speed_min/max.
    stop_probability: float = 0.1
    # Seconds after a speed ramp ends that still count as a speed transition
    # (EnvContext.steering.speed_transition). Read only by transition rewards.
    transition_hold_seconds: float = 1.0
    # > 0: after the hold, the transition also lasts until the trailing 1 s mean of
    # the speed along tar_dir is within this band of tar_speed. 0 keeps time only.
    transition_settle_band: float = 0.0


class ContinuousSteering(SteeringControl):
    def __init__(self, config, env):
        super().__init__(config, env)
        if not (0 <= config.fixed_fraction < 1 and 0 <= config.turn_fraction <= 1-config.fixed_fraction):
            raise ValueError("Invalid fixed/turn group fractions")
        if not (0 < config.acceleration_min <= config.acceleration_max and config.yaw_rate > 0):
            raise ValueError("Command rate limits must be positive")
        if not 0 <= config.independent_facing_fraction <= 1:
            raise ValueError("independent_facing_fraction must be in [0, 1]")
        self._goal_speed = torch.ones_like(self._tar_speed)
        self._goal_heading = torch.zeros_like(self._tar_dir_theta)
        self._acceleration = torch.ones_like(self._tar_speed) * config.acceleration_min
        self._mode = torch.zeros_like(self._heading_change_steps)
        self._independent_facing = torch.zeros_like(self._heading_change_steps, dtype=torch.bool)
        self._transition_steps_left = torch.zeros_like(self._heading_change_steps)
        self._speed_transition = torch.zeros_like(self._tar_speed, dtype=torch.bool)
        self._settling = torch.zeros_like(self._speed_transition)
        window = max(1, round(1.0 / env.dt))
        self._speed_history = torch.ones(len(self._tar_speed), window, device=self._tar_speed.device)
        self._history_index = 0

    def reset(self, env_ids):
        if len(env_ids) == 0:
            return
        root = self.env.simulator.get_root_state().root_pos[env_ids]
        self._prev_root_pos[env_ids] = root
        self._curr_root_pos[env_ids] = root
        r = torch.rand(len(env_ids), device=env_ids.device)
        self._mode[env_ids] = torch.where(r < self.config.fixed_fraction, 0,
            torch.where(r >= 1-self.config.turn_fraction, 2, 1))
        self._tar_speed[env_ids] = 1.
        self._tar_dir_theta[env_ids] = 0.
        self._tar_dir[env_ids] = torch.tensor([1., 0.], device=env_ids.device)
        self._tar_face_dir[env_ids] = self._tar_dir[env_ids]
        self._transition_steps_left[env_ids] = 0
        self._speed_transition[env_ids] = False
        self._settling[env_ids] = False
        self._resample_task(env_ids)
        fixed = env_ids[self._mode[env_ids] == 0]
        self._tar_speed[fixed] = self._goal_speed[fixed]
        self._speed_history[env_ids] = self._tar_speed[env_ids, None]

    def _resample_task(self, env_ids):
        if len(env_ids) == 0:
            return
        c = self.config
        n, device = len(env_ids), env_ids.device
        speed = torch.rand(n, device=device) * (c.tar_speed_max-c.tar_speed_min) + c.tar_speed_min
        speed = torch.where(torch.rand(n, device=device) < c.stop_probability, 0., speed)
        fixed = self._mode[env_ids] == 0
        # The fixed group retains its original command for the whole episode.
        initial = self.env.progress_buf[env_ids] == 0
        fixed_speed = torch.tensor([.8, 1., 1.2], device=device)[torch.randint(3, (n,), device=device)]
        speed = torch.where(fixed, torch.where(initial, fixed_speed, self._goal_speed[env_ids]), speed)
        if c.full_heading_for_turn:
            heading = (2 * torch.rand(n, device=device) - 1) * math.pi
        else:
            heading = self._tar_dir_theta[env_ids] + (2*torch.rand(n, device=device)-1)*c.turn_angle_max
        self._goal_heading[env_ids] = torch.where(self._mode[env_ids] == 2, heading, 0.)
        self._goal_speed[env_ids] = speed
        self._acceleration[env_ids] = c.acceleration_min + torch.rand(n, device=device)*(c.acceleration_max-c.acceleration_min)
        self._heading_change_steps[env_ids] = self.env.progress_buf[env_ids] + torch.randint(
            c.heading_change_steps_min, c.heading_change_steps_max, (n,), device=device)
        independent = (self._mode[env_ids] == 2) & (
            torch.rand(n, device=device) < c.independent_facing_fraction
        )
        self._independent_facing[env_ids] = independent
        face_angle = (2 * torch.rand(n, device=device) - 1) * math.pi
        facing = torch.stack((face_angle.cos(), face_angle.sin()), dim=-1)
        self._tar_face_dir[env_ids] = torch.where(
            independent[:, None], facing, self._tar_dir[env_ids]
        )

    def step(self):
        # Preserve position history even on command changes.
        self._prev_root_pos[:] = self._curr_root_pos
        self._curr_root_pos[:] = self.env.simulator.get_root_state().root_pos
        ids = (self.env.progress_buf >= self._heading_change_steps).nonzero(as_tuple=False).flatten()
        self._resample_task(ids)
        dv = self._acceleration * self.env.dt
        ramping = (self._goal_speed-self._tar_speed).abs() > 1e-6
        self._tar_speed += (self._goal_speed-self._tar_speed).clamp(-dv, dv)
        hold = round(self.config.transition_hold_seconds / self.env.dt)
        holding = self._transition_steps_left > 0
        self._speed_transition[:] = ramping | holding
        self._transition_steps_left[:] = torch.where(
            ramping, hold, (self._transition_steps_left-1).clamp(min=0))
        if self.config.transition_settle_band > 0:
            speed = ((self._curr_root_pos-self._prev_root_pos)[:, :2] / self.env.dt * self._tar_dir).sum(-1)
            fresh = self.env.progress_buf <= 1   # prev_root_pos was just reset: no speed yet
            self._speed_history[:, self._history_index] = torch.where(fresh, self._tar_speed, speed)
            self._history_index = (self._history_index + 1) % self._speed_history.shape[1]
            unsettled = (self._speed_history.mean(1) - self._tar_speed).abs() > self.config.transition_settle_band
            self._settling[:] = ramping | holding | (self._settling & unsettled)
            self._speed_transition |= self._settling
        angle_error = (self._goal_heading-self._tar_dir_theta+math.pi) % (2*math.pi)-math.pi
        max_turn = self.config.yaw_rate * self.env.dt
        self._tar_dir_theta += angle_error.clamp(-max_turn, max_turn)
        self._tar_dir[:, 0] = self._tar_dir_theta.cos()
        self._tar_dir[:, 1] = self._tar_dir_theta.sin()
        coupled = ~self._independent_facing
        self._tar_face_dir[coupled] = self._tar_dir[coupled]

    def populate_context(self, ctx):
        super().populate_context(ctx)
        ctx.steering.speed_transition = self._speed_transition
