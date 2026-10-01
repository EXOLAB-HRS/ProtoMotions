"""Behavior tests for the c-a7-cadence candidate (command, measurement, reward, labels)."""
from types import SimpleNamespace
import torch
from protomotions.envs.control.cadence_steering import CadenceSteering, CadenceSteeringConfig
from protomotions.envs.obs.cadence import compute_cadence_cond_from_motion_lib
from protomotions.envs.rewards.locomotion_quality import cadence_tracking, cadence_phase_contact

BODIES = ["Pelvis", "L_Ankle", "L_Toe", "R_Ankle", "R_Toe"]


def make(n, step_period):
    root = torch.zeros(n, 3)
    pos = torch.full((n, len(BODIES), 3), .2)
    env = SimpleNamespace(num_envs=n, device="cpu", dt=1/30,
        progress_buf=torch.zeros(n, dtype=torch.long),
        robot_config=SimpleNamespace(kinematic_info=SimpleNamespace(body_names=BODIES)),
        simulator=SimpleNamespace(get_root_state=lambda: SimpleNamespace(root_pos=root),
                                  get_robot_state=lambda: SimpleNamespace(rigid_body_pos=pos)))
    c = CadenceSteering(CadenceSteeringConfig(tar_speed_min=.5, tar_speed_max=1.5,
        heading_change_steps_min=90, heading_change_steps_max=181), env)
    c.reset(torch.arange(n))

    def walk(steps):
        for _ in range(steps):
            t = int(env.progress_buf[0]) + 1
            env.progress_buf += 1
            left = (t // step_period) % 2 == 0
            pos[..., 2] = .2
            pos[:, 1 if left else 3, 2] = .03
            c.step()
    return c, env, walk


def test_measured_cadence_matches_alternating_contacts():
    c, env, walk = make(4, step_period=18)
    walk(120)
    torch.testing.assert_close(c._measured, torch.full((4,), 30 / 18))
    assert (c._strikes >= 2).all()


def test_measured_cadence_decays_without_steps():
    c, env, walk = make(2, step_period=18)
    walk(60)
    before = c._measured.clone()
    for _ in range(60):
        env.progress_buf += 1
        c.step()
    assert (c._measured < before / 2).all()


def test_ratio_command_range_and_natural_share():
    torch.manual_seed(2944)
    c, env, walk = make(4000, step_period=18)
    c._resample_task(torch.arange(4000))
    walking = c._goal_speed > 0
    ratio = c._ratio[walking]
    assert ((ratio >= .9) & (ratio <= 1.1)).all()
    assert .25 < (ratio == 1).float().mean() < .35
    assert (c._ratio[~walking] == 1).all()


def test_cadence_reward_is_neutral_when_invalid():
    measured = torch.tensor([1.8, 1.8, 1.62])
    target = torch.tensor([1.8, 1.62, 1.8])
    valid = torch.tensor([True, True, False])
    r = cadence_tracking(measured, target, valid, relative_scale=.05)
    assert r[0] == 1 and r[2] == 1
    assert r[1] < .1


def test_expert_labels_follow_frame_index(tmp_path):
    labels = torch.tensor([1., .9, 1.1, 1., 1.05])
    path = tmp_path / "labels.pt"
    torch.save(dict(labels=labels, pack_frames=5), path)
    lib = SimpleNamespace(length_starts=torch.tensor([0, 3]), motion_num_frames=torch.tensor([3, 2]),
                          motion_dt=torch.tensor([.1, .1]))
    out = compute_cadence_cond_from_motion_lib(lib, torch.tensor([0, 0, 1, 1]),
                                               torch.tensor([.1, .5, 0., .1]), 1/30, str(path))
    torch.testing.assert_close(out[:, 0], torch.tensor([.9, 1.1, 1., 1.05]))


def make_phase(n, step_period):
    from protomotions.envs.control.cadence_steering import CadencePhaseSteering, CadencePhaseSteeringConfig
    root = torch.zeros(n, 3)
    pos = torch.full((n, len(BODIES), 3), .2)
    env = SimpleNamespace(num_envs=n, device="cpu", dt=1/30,
        progress_buf=torch.zeros(n, dtype=torch.long),
        robot_config=SimpleNamespace(kinematic_info=SimpleNamespace(body_names=BODIES)),
        simulator=SimpleNamespace(get_root_state=lambda: SimpleNamespace(root_pos=root),
                                  get_robot_state=lambda: SimpleNamespace(rigid_body_pos=pos)))
    c = CadencePhaseSteering(CadencePhaseSteeringConfig(tar_speed_min=.5, tar_speed_max=1.5,
        heading_change_steps_min=900, heading_change_steps_max=901), env)
    c.reset(torch.arange(n))
    c._tar_speed[:] = 1.0
    c._ratio[:] = 1.0

    def walk(steps):
        for _ in range(steps):
            t = int(env.progress_buf[0]) + 1
            env.progress_buf += 1
            left = (t // step_period) % 2 == 0
            pos[..., 2] = .2
            pos[:, 1 if left else 3, 2] = .03
            c.step()
    return c, walk


def test_clock_syncs_on_first_onset_and_runs_at_target_rate():
    c, walk = make_phase(2, step_period=18)
    assert not c._synced.any()
    walk(30)
    assert c._synced.all()
    before = c._phase.clone()
    walk(1)
    # The base class ramps _tar_speed inside step(); the clock uses the ramped value.
    f = c.target_ratio() * (c.config.cadence_f0_slope * c._tar_speed + c.config.cadence_f0_intercept)
    torch.testing.assert_close(torch.remainder(c._phase - before, 2 * torch.pi), torch.pi * f / 30)


def test_phase_reward_prefers_matching_stance_foot():
    contact = torch.tensor([[True, False], [False, True], [True, True], [False, True]])
    stance_left = torch.tensor([True, True, True, True])
    single = torch.tensor([True, True, True, False])
    r = cadence_phase_contact(contact, stance_left, single)
    torch.testing.assert_close(r, torch.tensor([1., 0., 1., 1.]))


def test_clock_keeps_phase_through_command_resample():
    c, walk = make_phase(2, step_period=18)
    walk(30)
    phase = c._phase.clone()
    c._resample_task(torch.arange(2))
    assert c._synced.all()
    torch.testing.assert_close(c._phase, phase)
