"""Candidate walking safeguards; engineering envelopes, not human population SD.

Cadence is steps/min (both feet). Step length is opposite-foot touchdown
separation projected onto commanded travel direction, not stride length.
"""
from dataclasses import dataclass
import torch
from protomotions.envs.control.steering_control import SteeringControl, SteeringControlConfig
from protomotions.utils.rotations import quaternion_to_matrix


def walking_envelope(speed):
    """Generous initial walk-only caps; validate against selected reference clips."""
    cadence_max = 100. + 25. * speed.clamp(0., 1.5)
    step_min = 60. * speed / cadence_max
    return cadence_max, step_min


def posture_angles(body_rot, body_pos, pelvis, chest, head):
    r = quaternion_to_matrix(body_rot, w_last=True)
    trunk = body_pos[:, chest] - body_pos[:, pelvis]
    # Heading-independent trunk inclination; head pitch is its forward axis vs horizontal.
    trunk_deg = torch.rad2deg(torch.atan2(torch.linalg.vector_norm(trunk[:, :2], dim=-1), trunk[:, 2]))
    head_pitch_deg = torch.rad2deg(torch.atan2(-r[:, head, 2, 0], torch.linalg.vector_norm(r[:, head, :2, 0], dim=-1)))
    return trunk_deg, head_pitch_deg


class StepTracker:
    """Debounced touchdown accounting, independent reset and stale-event detection."""
    def __init__(self, n, device, dt):
        self.dt = dt
        self.clock = torch.zeros(n, device=device)
        self.swing = torch.zeros(n, 2, device=device)
        self.last_touch = torch.full((n, 2), -100., device=device)
        self.last_pos = torch.zeros(n, 2, 2, device=device)
        self.last_side = torch.full((n,), -1, device=device, dtype=torch.long)
        self.last_event = torch.full((n,), -100., device=device)
        self.previous_contact = torch.ones(n, 2, device=device, dtype=torch.bool)
        self.cadence = torch.zeros(n, device=device)
        self.step_length = torch.zeros(n, device=device)
        self.events = torch.zeros(n, device=device, dtype=torch.long)

    def reset(self, ids):
        for tensor in (self.clock, self.swing, self.last_pos, self.cadence, self.step_length, self.events):
            tensor[ids] = 0
        self.last_touch[ids] = -100.; self.last_event[ids] = -100.
        self.last_side[ids] = -1; self.previous_contact[ids] = True

    def update(self, contact, foot_xy, direction):
        self.clock += self.dt
        touchdown = contact & ~self.previous_contact & (self.swing >= .10)
        self.swing = torch.where(contact, 0., self.swing + self.dt)
        # Simultaneous landings are hops, not alternating walking steps.
        single = touchdown.sum(-1) == 1
        for side in (0, 1):
            event = touchdown[:, side] & single
            alternating = event & (self.last_side == 1-side)
            elapsed = self.clock - self.last_event
            valid = alternating & (elapsed >= .15) & (elapsed <= 2.)
            length = ((foot_xy[:, side] - self.last_pos[:, 1-side]) * direction).sum(-1)
            self.cadence = torch.where(valid, 60./elapsed.clamp_min(.01), self.cadence)
            self.step_length = torch.where(valid, length, self.step_length)
            self.events += valid.long()
            self.last_touch[:, side] = torch.where(event, self.clock, self.last_touch[:, side])
            self.last_pos[:, side] = torch.where(event[:, None], foot_xy[:, side], self.last_pos[:, side])
            self.last_event = torch.where(event, self.clock, self.last_event)
            self.last_side = torch.where(event, side, self.last_side)
        self.previous_contact = contact.clone()
        return (self.events >= 2) & (self.clock - self.last_event < 1.5)


def walking_penalty(penalty):
    return penalty


@dataclass
class WalkingSteeringConfig(SteeringControlConfig):
    _target_: str = 'protomotions.envs.control.walking_quality.WalkingSteeringControl'
    transition_grace_s: float = 2.


class WalkingSteeringControl(SteeringControl):
    """Steering plus measured gait quality; no direct target-angle animation."""
    def __init__(self, config, env):
        super().__init__(config, env)
        names = env.robot_config.kinematic_info.body_names
        self.feet = [[names.index(s+'_Ankle'), names.index(s+'_Toe')] for s in ('L','R')]
        self.ankles = [names.index(s+'_Ankle') for s in ('L','R')]
        self.posture_ids = [names.index(x) for x in ('Pelvis','Chest','Head')]
        self.tracker = StepTracker(env.num_envs, env.device, env.dt)
        self.stable_time = torch.zeros(env.num_envs, device=env.device)
        self.quality_penalty = torch.zeros_like(self.stable_time)
        self.trunk_deg = torch.zeros_like(self.stable_time)
        self.head_pitch_deg = torch.zeros_like(self.stable_time)
        self.slip_mps = torch.zeros_like(self.stable_time)
        self.quality_valid = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    def reset(self, ids):
        super().reset(ids)
        self.tracker.reset(ids)
        for x in (self.stable_time,self.quality_penalty,self.trunk_deg,self.head_pitch_deg,self.slip_mps):x[ids]=0
        self.quality_valid[ids]=False

    def step(self):
        old_speed, old_dir = self._tar_speed.clone(), self._tar_dir.clone()
        super().step()
        changed = ((old_speed-self._tar_speed).abs()>.05) | ((old_dir*self._tar_dir).sum(-1)<.995)
        self.stable_time = torch.where(changed,0.,self.stable_time+self.env.dt)
        state = self.env.simulator.get_robot_state()
        forces = self.env.simulator.get_bodies_contact_buf().rigid_body_contact_forces
        contact = torch.stack([torch.linalg.vector_norm(forces[:,ids],dim=-1).sum(-1)>10 for ids in self.feet],-1)
        valid = self.tracker.update(contact,state.rigid_body_pos[:,self.ankles,:2],self._tar_dir)
        self.trunk_deg,self.head_pitch_deg = posture_angles(state.rigid_body_rot,state.rigid_body_pos,*self.posture_ids)
        foot_speed = torch.stack([torch.linalg.vector_norm(state.rigid_body_vel[:,ids,:2],dim=-1).mean(-1) for ids in self.feet],-1)
        self.slip_mps = (foot_speed*contact).sum(-1)/contact.sum(-1).clamp_min(1)
        moving = self._tar_speed>.3
        forward = (self._tar_dir*self._tar_face_dir).sum(-1)>.85
        steady = self.stable_time>self.config.transition_grace_s
        cap,length_min = walking_envelope(self._tar_speed)
        # Quality does not confer a positive reward for standing instead of walking.
        cadence_cost = ((self.tracker.cadence-cap).clamp_min(0)/20).square()
        short_cost = ((length_min-self.tracker.step_length).clamp_min(0)/.15).square()
        self.quality_valid = valid & moving & steady & forward
        gait_cost = torch.where(self.quality_valid,cadence_cost+short_cost,0.)
        stale_cost = (moving & steady & ~valid).float()
        lean_limit = torch.where(steady,10.,20.)
        posture_cost = ((self.trunk_deg-lean_limit).clamp_min(0)/10).square()
        posture_cost += ((self.head_pitch_deg.abs()-15).clamp_min(0)/10).square()
        self.quality_penalty = gait_cost.clamp_max(10)+stale_cost+posture_cost.clamp_max(10)+2*self.slip_mps.square().clamp_max(2)

        self.env.extras.update({
            'walking/cadence_spm':self.tracker.cadence,
            'walking/step_length_m':self.tracker.step_length,
            'walking/valid':self.quality_valid.float(),
            'walking/trunk_deg':self.trunk_deg,
            'walking/head_pitch_deg':self.head_pitch_deg,
            'walking/stance_slip_mps':self.slip_mps,
        })

    def check_resets_and_terminations(self):
        state = self.env.simulator.get_robot_state()
        ground = self.env.terrain.get_ground_heights(state.rigid_body_pos[:,0]).reshape(-1)
        height = state.rigid_body_pos[:,0,2] - ground
        fallen = (height < .55) | (self.trunk_deg > 60.)
        finite = torch.isfinite(state.dof_pos).all(-1) & torch.isfinite(state.dof_vel).all(-1)
        terminate = ((self.env.progress_buf > 10) & fallen) | ~finite
        return terminate.clone(), terminate

    def populate_context(self, ctx):
        super().populate_context(ctx)
        ctx.steering.walking_quality_penalty = self.quality_penalty


def evaluate_walking_trace(body_pos, body_rot, body_vel, contacts, target_speed,
                           target_direction, body_names, dt, warmup_s=2.):
    """Independent per-rollout gates. Missing steps/coverage never count as pass.

    Inputs are [time,body,...]; caller must supply steady forward-walking trials.
    Simulator contact forces must be thresholded, rather than inferred from height.
    """
    if dt <= 0 or warmup_s < 0:
        raise ValueError('dt must be positive and warmup_s nonnegative')
    if len(body_pos) * dt <= warmup_s:
        return dict(metrics={'speed_command_mps': None, 'duration_s': len(body_pos)*dt},
                    checks={'duration': False}, status='candidate_failed_gate',
                    scope='No post-warmup samples')
    ids=[body_names.index(x) for x in ('Pelvis','Chest','Head')]
    ankles=[body_names.index(s+'_Ankle') for s in ('L','R')]
    foot_ids=[[body_names.index(s+'_Ankle'),body_names.index(s+'_Toe')] for s in ('L','R')]
    tracker=StepTracker(1,body_pos.device,dt)
    cadence=[];lengths=[];slips=[]
    for t in range(len(body_pos)):
        contact=torch.stack([contacts[t,f].bool().any() for f in foot_ids])[None]
        valid=tracker.update(contact,body_pos[t,ankles,:2][None],target_direction[t:t+1])
        if t*dt>=warmup_s:
            if valid[0]:cadence.append(tracker.cadence[0].clone());lengths.append(tracker.step_length[0].clone())
            for side,f in enumerate(foot_ids):
                if contact[0,side]:slips.extend(torch.linalg.vector_norm(body_vel[t,f,:2],dim=-1).unbind())
    start=int(warmup_s/dt)
    trunk,head=posture_angles(body_rot,body_pos,*ids)
    velocity=body_vel[start:,0,:2]
    requested=target_speed[start:,None]*target_direction[start:]
    mean_speed=target_speed[start:].mean()
    cap,min_length=walking_envelope(mean_speed)
    def quantile(values,p):return float(torch.quantile(torch.stack(values),p)) if values else None
    metrics=dict(speed_command_mps=float(mean_speed),velocity_mae_mps=float(torch.linalg.vector_norm(velocity-requested,dim=-1).mean()),
        cadence_p95_spm=quantile(cadence,.95),cadence_median_spm=quantile(cadence,.5),step_length_median_m=quantile(lengths,.5),
        trunk_p95_deg=float(torch.quantile(trunk[start:],.95)),head_pitch_abs_p95_deg=float(torch.quantile(head[start:].abs(),.95)),
        slip_p95_mps=quantile(slips,.95),alternating_events=int(tracker.events[0]),duration_s=len(body_pos)*dt)
    checks=dict(duration=metrics['duration_s']>=8.,events=metrics['alternating_events']>=4,
        velocity=metrics['velocity_mae_mps']<=.15,cadence=metrics['cadence_p95_spm'] is not None and metrics['cadence_p95_spm']<=float(cap),
        step_length=metrics['step_length_median_m'] is not None and metrics['step_length_median_m']>=float(min_length),
        trunk=metrics['trunk_p95_deg']<=15.,head=metrics['head_pitch_abs_p95_deg']<=20.,
        slip=metrics['slip_p95_mps'] is not None and metrics['slip_p95_mps']<=.2)
    return dict(metrics=metrics,checks=checks,status='passed' if all(checks.values()) else 'candidate_failed_gate',scope='Candidate engineering thresholds; not normative population bands')


def evaluate_speed_sweep(reports):
    """Require all five forward speeds and increasing step length, not cadence alone."""
    by_speed={round(r['metrics']['speed_command_mps'],2):r for r in reports
              if r['metrics'].get('speed_command_mps') is not None}
    coverage=all(s in by_speed for s in (.5,.75,1.,1.25,1.5))
    checks={'coverage':coverage,'individual_gates':bool(reports) and all(r['status']=='passed' for r in reports)}
    if coverage:
        slow,fast=by_speed[.5]['metrics'],by_speed[1.5]['metrics']
        checks['step_length_increases']=slow['step_length_median_m'] is not None and fast['step_length_median_m'] is not None and fast['step_length_median_m']>=1.3*slow['step_length_median_m']
        checks['cadence_not_only_strategy']=slow['cadence_median_spm'] is not None and fast['cadence_median_spm'] is not None and fast['cadence_median_spm']<=1.8*slow['cadence_median_spm']
    return dict(checks=checks,status='passed' if all(checks.values()) else 'candidate_failed_gate',scope='Forward steady walk only; acceleration/turning/backward/stop are separate trials')
