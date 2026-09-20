import torch

from protomotions.envs.rewards.task import compute_neck_stability_rew


NECK_HEAD_IDS = [23, 24, 25, 26, 27, 28]
REFERENCE_P95 = [
    0.7622655034,
    1.4256707430,
    0.8655657172,
    0.4837467074,
    0.4921766818,
    0.4224115610,
]


def test_neck_stability_reward_is_one_inside_reference_envelope():
    dof_vel = torch.zeros(2, 59)
    dof_vel[1, NECK_HEAD_IDS] = torch.tensor(REFERENCE_P95)

    reward = compute_neck_stability_rew(
        dof_vel, NECK_HEAD_IDS, REFERENCE_P95
    )

    torch.testing.assert_close(reward, torch.ones(2))


def test_neck_stability_reward_penalizes_only_excess_neck_head_speed():
    dof_vel = torch.zeros(3, 59)
    dof_vel[0, 0] = 100.0
    dof_vel[1, 23] = 2.0 * REFERENCE_P95[0]
    dof_vel[2, 23] = -2.0 * REFERENCE_P95[0]

    reward = compute_neck_stability_rew(
        dof_vel, NECK_HEAD_IDS, REFERENCE_P95
    )

    torch.testing.assert_close(reward[0], torch.tensor(1.0))
    assert 0.0 < reward[1] < 1.0
    torch.testing.assert_close(reward[1], reward[2])
