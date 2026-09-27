"""Behavior tests for a9/a10 rewards and b1-a7 command continuity."""
import math
from types import SimpleNamespace
import torch
import pytest
from protomotions.envs.rewards.locomotion_quality import (
    head_upright, head_angular_stability, reference_bounded_head_motion,
    target_second_difference,
    speed_transition_tracking,
)
from protomotions.envs.control.continuous_steering import ContinuousSteering, ContinuousSteeringConfig


def test_overspeed_transition_penalty_preserves_other_rewards():
    current = torch.tensor([[0.9, 0., 0.], [1.1, 0., 0.], [1.1, 0., 0.]])
    previous = torch.zeros_like(current)
    direction = torch.tensor([[1., 0.]]).expand(3, 2)
    target = torch.ones(3)
    transition = torch.tensor([True, True, False])
    original = speed_transition_tracking(current, previous, direction, target, transition, 1., 20.)
    candidate = speed_transition_tracking(current, previous, direction, target, transition, 1., 20., 80.)
    torch.testing.assert_close(candidate[[0, 2]], original[[0, 2]])
    assert candidate[1] < original[1]


def test_overspeed_tail_keeps_gradient_at_large_error():
    current = torch.tensor([[1.3, 0., 0.], [0.7, 0., 0.], [1.3, 0., 0.]])
    previous = torch.zeros_like(current)
    direction = torch.tensor([[1., 0.]]).expand(3, 2)
    target = torch.ones(3)
    transition = torch.tensor([True, True, False])
    base = speed_transition_tracking(current, previous, direction, target, transition, 1., 20., 80.)
    tail = speed_transition_tracking(current, previous, direction, target, transition, 1.,
                                     20., 80., 8., 0.5)
    assert tail[0] > base[0] + 0.2
    torch.testing.assert_close(tail[1:], base[1:])
    near = speed_transition_tracking(torch.tensor([[1.05, 0., 0.]]), torch.zeros(1, 3),
                                     direction[:1], target[:1], transition[:1], 1.,
                                     20., 80., 8., 0.5)
    assert near[0] < 0.95


def test_underspeed_tail_keeps_restart_gradient_without_changing_braking():
    current = torch.tensor([[.2, 0., 0.], [1.4, 0., 0.], [.2, 0., 0.]])
    previous = torch.zeros_like(current)
    direction = torch.tensor([[1., 0.]]).expand(3, 2)
    target = torch.ones(3)
    transition = torch.tensor([True, True, False])
    base = speed_transition_tracking(current, previous, direction, target, transition, 1., 20., 40.)
    tailed = speed_transition_tracking(current, previous, direction, target, transition, 1.,
                                       20., 40., 8., .5, 4., .5)
    assert tailed[0] > base[0] + .02
    assert tailed[1] > base[1] + .1
    assert tailed[2] == base[2] == 1.
    current = torch.tensor([[.2, 0., 0.]], requires_grad=True)
    reward = speed_transition_tracking(current, torch.zeros_like(current), direction[:1],
                                       target[:1], transition[:1], 1., 20., 40.,
                                       8., .5, 4., .5)
    reward.sum().backward()
    assert current.grad[0, 0] > .01


def test_head_reward_rejects_static_tilt_but_allows_yaw():
    q = torch.tensor([[[0.,0.,0.,1.]], [[0.,0.,1.,0.]],
                      [[math.sin(math.pi/36),0.,0.,math.cos(math.pi/36)]], [[1.,0.,0.,0.]]])
    r = head_upright(q, 0)
    torch.testing.assert_close(r[:2], torch.ones(2))
    assert 0 < r[2] < .5 and r[3] < .001
    assert torch.equal(head_upright(q,0), head_upright(-q,0))


def test_head_velocity_ignores_yaw_only():
    w = torch.tensor([[[0.,0.,10.]], [[1.,0.,0.]], [[0.,1.,0.]]])
    r = head_angular_stability(w, 0)
    assert r[0] == 1 and r[1] == r[2] and r[1] < .5


def test_target_reward_preserves_constant_rate_and_penalizes_reversal():
    current = torch.tensor([[.2,.4],[0.,0.]])
    history = torch.tensor([[[.1,.2],[0.,0.]], [[.1,.2],[0.,0.]]])
    r = target_second_difference(current, history)
    assert r[0] == 1 and r[1] < .01


def test_reference_bounded_head_motion_penalizes_only_envelope_excess():
    identity = torch.tensor([0., 0., 0., 1.])
    tilted = torch.tensor([math.sin(math.pi / 8), 0., 0., math.cos(math.pi / 8)])
    rotations = torch.stack((
        torch.stack((identity, identity)),
        torch.stack((identity, tilted)),
    ))
    angular_velocity = torch.tensor([
        [[0., 0., 2.], [0., 0., 2.]],
        [[0., 0., 0.], [3., 0., 0.]],
    ])
    reward = reference_bounded_head_motion(
        rotations, angular_velocity, head_index=1, chest_index=0,
        tilt_limit_rad=math.radians(15.7145),
        relative_speed_limit_rad_s=math.radians(77.2072),
    )
    assert reward[0] == 1
    assert reward[1] < 0.1


def test_continuous_commands_bound_rates_keep_velocity_and_mix_modes():
    torch.manual_seed(2944)
    root = torch.zeros(600,3)
    env = SimpleNamespace(num_envs=600, device='cpu', dt=1/30,
        progress_buf=torch.zeros(600,dtype=torch.long),
        simulator=SimpleNamespace(get_root_state=lambda:SimpleNamespace(root_pos=root)))
    c = ContinuousSteering(ContinuousSteeringConfig(tar_speed_min=.5,tar_speed_max=1.5,
        heading_change_steps_min=4,heading_change_steps_max=5),env)
    c.reset(torch.arange(600));fixed=c._mode==0;fixed_speed=c._tar_speed[fixed].clone()
    assert fixed.any() and (c._mode==1).any() and (c._mode==2).any()
    for _ in range(200):
        old_speed,old_angle=c._tar_speed.clone(),c._tar_dir_theta.clone()
        root[:,0] += env.dt;env.progress_buf+=1;c.step()
        assert ((c._tar_speed-old_speed).abs() <= .8/30+1e-6).all()
        assert ((c._tar_dir_theta-old_angle).abs() <= .35/30+1e-6).all()
        torch.testing.assert_close(root-c._prev_root_pos,torch.tensor([env.dt,0.,0.]).expand(600,3),atol=1e-6,rtol=0)
        torch.testing.assert_close(c._tar_speed[fixed],fixed_speed)
        torch.testing.assert_close(c._tar_face_dir,c._tar_dir)
        assert ((c._tar_speed >= 0) & (c._tar_speed <= 1.5)).all()


def test_speed_anchor_sampling_keeps_continuous_coverage():
    torch.manual_seed(2944)
    root = torch.zeros(2000, 3)
    env = SimpleNamespace(num_envs=2000, device='cpu', dt=1/30,
        progress_buf=torch.zeros(2000, dtype=torch.long),
        simulator=SimpleNamespace(get_root_state=lambda: SimpleNamespace(root_pos=root)))
    c = ContinuousSteering(ContinuousSteeringConfig(
        tar_speed_min=.2, tar_speed_max=1.4, fixed_fraction=0., turn_fraction=0.,
        stop_probability=0., speed_anchor_targets=(.4, 1., 1.4),
        speed_anchor_probability=.7), env)
    c.reset(torch.arange(2000))
    targets = c._goal_speed
    anchored = torch.isin(targets, torch.tensor([.4, 1., 1.4]))
    assert .65 < anchored.float().mean() < .75
    for value in (.4, 1., 1.4):
        assert (targets == value).sum() > 400
    assert ((targets[~anchored] >= .2) & (targets[~anchored] <= 1.4)).all()


@pytest.mark.parametrize('options', [
    dict(speed_anchor_probability=1., speed_anchor_targets=None),
    dict(speed_anchor_probability=1., speed_anchor_targets=(2.1,)),
    dict(speed_anchor_probability=-.1),
])
def test_speed_anchor_rejects_invalid_settings(options):
    env = SimpleNamespace(num_envs=1, device='cpu', dt=1/30)
    with pytest.raises(ValueError):
        ContinuousSteering(ContinuousSteeringConfig(**options), env)


def test_acceleration_anchor_sampling_keeps_continuous_coverage():
    torch.manual_seed(2944)
    root = torch.zeros(2000, 3)
    env = SimpleNamespace(num_envs=2000, device='cpu', dt=1/30,
        progress_buf=torch.zeros(2000, dtype=torch.long),
        simulator=SimpleNamespace(get_root_state=lambda: SimpleNamespace(root_pos=root)))
    c = ContinuousSteering(ContinuousSteeringConfig(
        tar_speed_min=.2, tar_speed_max=1.4, fixed_fraction=0., turn_fraction=0.,
        acceleration_min=.4, acceleration_max=.8, stop_probability=0.,
        acceleration_anchor_targets=(.4, .8), acceleration_anchor_probability=.7), env)
    c.reset(torch.arange(2000))
    rates = c._acceleration
    anchored = torch.isin(rates, torch.tensor([.4, .8]))
    assert .65 < anchored.float().mean() < .75
    assert (rates == .4).sum() > 500 and (rates == .8).sum() > 500
    assert ((rates[~anchored] >= .4) & (rates[~anchored] <= .8)).all()


def test_transition_pair_sampling_emphasizes_high_speed_braking_and_keeps_stops():
    torch.manual_seed(2944)
    root = torch.zeros(2000, 3)
    env = SimpleNamespace(num_envs=2000, device='cpu', dt=1/30,
        progress_buf=torch.ones(2000, dtype=torch.long),
        simulator=SimpleNamespace(get_root_state=lambda: SimpleNamespace(root_pos=root)))
    c = ContinuousSteering(ContinuousSteeringConfig(
        tar_speed_min=.2, tar_speed_max=1.4, fixed_fraction=0., turn_fraction=0.,
        stop_probability=.3, transition_pair_targets=((1.4, 1.0),),
        transition_pair_probability=.7), env)
    c._mode[:] = 1
    c._tar_speed[:] = 1.4
    c._resample_task(torch.arange(2000))
    targets = c._goal_speed
    assert .27 < (targets == 0).float().mean() < .33
    assert (targets[targets > 0] == 1.0).float().mean() > .65
    assert ((targets > 0) & (targets != 1.0)).any()
    c._tar_speed[:] = 1.2
    c._resample_task(torch.arange(2000))
    assert (c._goal_speed[c._goal_speed > 0] == 1.0).float().mean() < .3


@pytest.mark.parametrize('options', [
    dict(transition_pair_probability=1., transition_pair_targets=None),
    dict(transition_pair_probability=1., transition_pair_targets=((1.4, 1.4),)),
    dict(transition_pair_probability=1., transition_pair_targets=((1.4, 10.),)),
    dict(transition_pair_probability=-.1),
    dict(transition_pair_source_band=0.),
])
def test_transition_pair_rejects_invalid_settings(options):
    env = SimpleNamespace(num_envs=1, device='cpu', dt=1/30)
    with pytest.raises(ValueError):
        ContinuousSteering(ContinuousSteeringConfig(**options), env)


def test_command_preview_is_bounded_and_keeps_current_reward_target():
    env = SimpleNamespace(num_envs=2, device='cpu', dt=1/30)
    c = ContinuousSteering(ContinuousSteeringConfig(
        preview_seconds=.5, acceleration_min=.4, acceleration_max=.8,
        yaw_rate=.5), env)
    c._tar_speed[:] = torch.tensor([1.0, 1.0])
    c._goal_speed[:] = torch.tensor([0.0, 1.4])
    c._acceleration[:] = torch.tensor([.4, .8])
    c._goal_heading[:] = torch.tensor([math.pi / 2, 0.0])
    c._tar_dir_theta[:] = 0
    ctx = SimpleNamespace(steering=None)
    c.populate_context(ctx)
    torch.testing.assert_close(ctx.steering.tar_speed, torch.ones(2))
    torch.testing.assert_close(ctx.steering.tar_speed_preview, torch.tensor([.8, 1.4]))
    torch.testing.assert_close(ctx.steering.tar_dir_preview[0],
                               torch.tensor([math.cos(.25), math.sin(.25)]))
    torch.testing.assert_close(ctx.steering.tar_dir, torch.tensor([[1., 0.], [1., 0.]]))
    torch.testing.assert_close(ctx.steering.tar_face_dir_preview,
                               ctx.steering.tar_dir_preview)
    c.config.preview_seconds = 0
    c.populate_context(ctx)
    assert ctx.steering.tar_speed_preview is ctx.steering.tar_speed
    assert ctx.steering.tar_dir_preview is ctx.steering.tar_dir


def test_command_preview_rejects_invalid_horizon():
    env = SimpleNamespace(num_envs=1, device='cpu', dt=1/30)
    for horizon in (-.1, float('nan'), float('inf')):
        with pytest.raises(ValueError, match='preview_seconds'):
            ContinuousSteering(ContinuousSteeringConfig(preview_seconds=horizon), env)


def test_command_preview_uses_shorter_horizon_only_for_acceleration():
    env = SimpleNamespace(num_envs=2, device='cpu', dt=1/30)
    c = ContinuousSteering(ContinuousSteeringConfig(
        preview_seconds=1.0, acceleration_preview_seconds=.5,
        acceleration_min=.4, acceleration_max=.8, yaw_rate=.5), env)
    c._tar_speed[:] = 1.0
    c._goal_speed[:] = torch.tensor([0.0, 2.0])
    c._acceleration[:] = torch.tensor([.4, .8])
    c._goal_heading[:] = math.pi / 2
    ctx = SimpleNamespace(steering=None)
    c.populate_context(ctx)
    torch.testing.assert_close(ctx.steering.tar_speed_preview, torch.tensor([.6, 1.4]))
    torch.testing.assert_close(ctx.steering.tar_dir_preview[0],
                               torch.tensor([math.cos(.5), math.sin(.5)]))
    with pytest.raises(ValueError, match='acceleration_preview_seconds'):
        ContinuousSteering(ContinuousSteeringConfig(acceleration_preview_seconds=-.5), env)


def test_command_preview_keeps_long_stop_but_shortens_walking_brake():
    env = SimpleNamespace(num_envs=3, device='cpu', dt=1/30)
    c = ContinuousSteering(ContinuousSteeringConfig(
        preview_seconds=1.0, acceleration_preview_seconds=.5,
        nonstop_deceleration_preview_seconds=.25, acceleration_min=.4,
        acceleration_max=.8, yaw_rate=.5), env)
    c._tar_speed[:] = torch.tensor([1.4, 1.0, 1.0])
    c._goal_speed[:] = torch.tensor([1.0, 0.0, 1.4])
    c._acceleration[:] = torch.tensor([.4, .4, .8])
    ctx = SimpleNamespace(steering=None)
    c.populate_context(ctx)
    torch.testing.assert_close(ctx.steering.tar_speed_preview, torch.tensor([1.3, .6, 1.4]))
    torch.testing.assert_close(ctx.steering.tar_speed, torch.tensor([1.4, 1.0, 1.0]))
    with pytest.raises(ValueError, match='nonstop_deceleration_preview_seconds'):
        ContinuousSteering(ContinuousSteeringConfig(nonstop_deceleration_preview_seconds=-.1), env)


@pytest.mark.parametrize('options', [
    dict(acceleration_anchor_probability=1., acceleration_anchor_targets=None),
    dict(acceleration_anchor_probability=1., acceleration_anchor_targets=(1.0,)),
    dict(acceleration_anchor_probability=-.1),
])
def test_acceleration_anchor_rejects_invalid_settings(options):
    env = SimpleNamespace(num_envs=1, device='cpu', dt=1/30)
    with pytest.raises(ValueError):
        ContinuousSteering(ContinuousSteeringConfig(**options), env)


def test_slow_turn_commands_replay_straight_and_stop_groups():
    torch.manual_seed(2944)
    root = torch.zeros(1000, 3)
    env = SimpleNamespace(num_envs=1000, device='cpu', dt=1/30,
        progress_buf=torch.zeros(1000, dtype=torch.long),
        simulator=SimpleNamespace(get_root_state=lambda: SimpleNamespace(root_pos=root)))
    c = ContinuousSteering(ContinuousSteeringConfig(
        fixed_fraction=.3, turn_fraction=.2, yaw_rate=.5,
        heading_change_steps_min=2, heading_change_steps_max=3,
        turn_angle_max=math.pi/2, stop_probability=.2), env)
    c.reset(torch.arange(1000))
    assert .25 < (c._mode == 0).float().mean() < .35
    assert .15 < (c._mode == 2).float().mean() < .25
    for _ in range(90):
        previous = c._tar_dir_theta.clone()
        env.progress_buf += 1
        c.step()
        assert (c._tar_dir_theta - previous).abs().max() <= .5/30 + 1e-6
        torch.testing.assert_close(c._tar_face_dir, c._tar_dir)
    assert (c._tar_dir_theta[c._mode == 2].abs() > .2).any()
    assert torch.equal(c._tar_dir_theta[c._mode != 2], torch.zeros_like(c._tar_dir_theta[c._mode != 2]))


@pytest.mark.parametrize('fraction', [.5, 1.0])
def test_stage_d_continuous_commands_cover_full_heading_and_independent_facing(fraction):
    torch.manual_seed(2947)
    root = torch.zeros(1000, 3)
    env = SimpleNamespace(num_envs=1000, device='cpu', dt=1/30,
        progress_buf=torch.zeros(1000, dtype=torch.long),
        simulator=SimpleNamespace(get_root_state=lambda: SimpleNamespace(root_pos=root)))
    config = ContinuousSteeringConfig(
        fixed_fraction=0.2, turn_fraction=0.6,
        full_heading_for_turn=True, independent_facing_fraction=fraction,
        heading_change_steps_min=4, heading_change_steps_max=5,
    )
    control = ContinuousSteering(config, env)
    control.reset(torch.arange(1000))
    turning = control._mode == 2
    assert turning.sum() > 500
    assert (control._goal_heading[turning].abs() > math.pi / 2).any()
    facing_alignment = (control._tar_face_dir[turning] * control._tar_dir[turning]).sum(-1)
    assert (facing_alignment < 0.5).any()
    assert abs(control._independent_facing[turning].float().mean().item() - fraction) < .08
    assert not control._independent_facing[~turning].any()
    fixed = control._mode == 0
    assert torch.isin(control._tar_speed[fixed], torch.tensor([.8, 1., 1.2])).all()
    # Re-sampling must retain the aligned replay group while allowing the D
    # group to point its travel and facing commands in different directions.
    env.progress_buf += 5
    control.step()
    coupled = ~control._independent_facing
    torch.testing.assert_close(control._tar_face_dir[coupled], control._tar_dir[coupled])


def test_relative_facing_bounds_rates_resampling_and_partial_reset():
    torch.manual_seed(2944)
    root = torch.zeros(1000, 3)
    env = SimpleNamespace(num_envs=1000, device='cpu', dt=1/30,
        progress_buf=torch.zeros(1000, dtype=torch.long),
        simulator=SimpleNamespace(get_root_state=lambda: SimpleNamespace(root_pos=root)))
    cfg = ContinuousSteeringConfig(fixed_fraction=.3, turn_fraction=.3,
        yaw_rate=1., turn_angle_max=math.pi/4, independent_facing_fraction=1.,
        facing_offset_max=math.pi/12, facing_yaw_rate=.5,
        heading_change_steps_min=90, heading_change_steps_max=181)
    c = ContinuousSteering(cfg, env)
    c.reset(torch.arange(1000))
    wrap = lambda x: (x+math.pi) % (2*math.pi)-math.pi
    assert abs((c._mode == 2).float().mean().item()-.3) < .05
    for _ in range(600):
        old_face = torch.atan2(c._tar_face_dir[:, 1], c._tar_face_dir[:, 0])
        old_travel = c._tar_dir_theta.clone()
        env.progress_buf += 1
        c.step()
        face = torch.atan2(c._tar_face_dir[:, 1], c._tar_face_dir[:, 0])
        assert wrap(face-old_face).abs().max() <= .5/30+2e-6
        assert wrap(c._tar_dir_theta-old_travel).abs().max() <= 1/30+2e-6
        assert wrap(face-c._tar_dir_theta).abs().max() <= math.pi/12+2e-6
    before = c._tar_face_dir[10:].clone()
    env.progress_buf[:10] = 0
    c.reset(torch.arange(10))
    torch.testing.assert_close(c._tar_face_dir[10:], before)
    torch.testing.assert_close(c._tar_face_dir[:10], torch.tensor([1., 0.]).expand(10, 2))
    # Cross +pi toward -pi along the short arc, reaching the requested offset.
    c._tar_dir_theta[:] = math.radians(179)
    c._tar_face_dir[:] = torch.tensor([math.cos(math.radians(179)), math.sin(math.radians(179))])
    c._goal_heading[:] = math.radians(-179)
    c._goal_facing[:] = math.radians(-169)
    c._heading_change_steps[:] = 100000
    for _ in range(60):
        c.step()
    face = torch.atan2(c._tar_face_dir[:, 1], c._tar_face_dir[:, 0])
    torch.testing.assert_close(wrap(face-math.radians(-169)), torch.zeros(1000), atol=2e-6, rtol=0)
    torch.testing.assert_close(c._tar_dir_theta, torch.full((1000,), math.radians(181)), atol=2e-6, rtol=0)


@pytest.mark.parametrize('options', [dict(facing_yaw_rate=0), dict(facing_yaw_rate=float('nan')),
    dict(facing_offset_max=math.pi), dict(full_heading_for_turn=True)])
def test_relative_facing_rejects_invalid_config(options):
    env = SimpleNamespace(num_envs=1, device='cpu', dt=1/30)
    with pytest.raises(ValueError):
        ContinuousSteering(ContinuousSteeringConfig(**{'facing_offset_max': math.pi/12, **options}), env)
