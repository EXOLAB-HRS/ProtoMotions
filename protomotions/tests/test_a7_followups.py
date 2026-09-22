"""Behavior tests for a9/a10 rewards and b1-a7 command continuity."""
import math
from types import SimpleNamespace
import torch
from protomotions.envs.rewards.locomotion_quality import (
    head_upright, head_angular_stability, reference_bounded_head_motion,
    target_second_difference,
)
from protomotions.envs.control.continuous_steering import ContinuousSteering, ContinuousSteeringConfig


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


def test_stage_d_continuous_commands_cover_full_heading_and_independent_facing():
    torch.manual_seed(2947)
    root = torch.zeros(1000, 3)
    env = SimpleNamespace(num_envs=1000, device='cpu', dt=1/30,
        progress_buf=torch.zeros(1000, dtype=torch.long),
        simulator=SimpleNamespace(get_root_state=lambda: SimpleNamespace(root_pos=root)))
    config = ContinuousSteeringConfig(
        fixed_fraction=0.2, turn_fraction=0.6,
        full_heading_for_turn=True, independent_facing_fraction=1.0,
        heading_change_steps_min=4, heading_change_steps_max=5,
    )
    control = ContinuousSteering(config, env)
    control.reset(torch.arange(1000))
    turning = control._mode == 2
    assert turning.sum() > 500
    assert (control._goal_heading[turning].abs() > math.pi / 2).any()
    facing_alignment = (control._tar_face_dir[turning] * control._tar_dir[turning]).sum(-1)
    assert (facing_alignment < 0.5).any()
