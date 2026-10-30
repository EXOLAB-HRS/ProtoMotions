"""2026-10-30: AMP discriminator replay fix (keep_prob once full) and first-episode timeout desync."""
from types import SimpleNamespace

import torch

from protomotions.agents.amp.agent import AMP
from protomotions.agents.utils.replay_buffer import ReplayBuffer
from protomotions.envs.base_env.env import BaseEnv


def _agent(fix):
    params = SimpleNamespace(discriminator_replay_keep_prob=0.01, discriminator_replay_fix=fix)
    return SimpleNamespace(amp_replay_buffer=ReplayBuffer(1000, torch.device("cpu")), device="cpu",
                           config=SimpleNamespace(amp_parameters=params))


def _fill(agent, epochs, n=600):
    for e in range(epochs):
        AMP.update_disc_replay_buffer(agent, {"x": torch.full((n, 1), float(e))})
    return agent.amp_replay_buffer.x[:, 0]


def test_old_condition_overwrites_everything():
    x = _fill(_agent(False), 6)
    assert set(x.tolist()) <= {4.0, 5.0}          # only the newest ~1.7 epochs remain


def test_fix_keeps_old_samples_once_full():
    torch.manual_seed(0)
    x = _fill(_agent(True), 6)
    assert (x <= 1.0).float().mean() > 0.9         # after the buffer filled, only ~1 % of new samples enter


def test_desync_first_timeouts_range_and_reset():
    env = SimpleNamespace(num_envs=4096, max_episode_length=300, device="cpu")
    BaseEnv.desync_first_timeouts(env)
    lim = env._first_episode_limit
    assert lim.min() >= 1 and lim.max() <= 300 and lim.float().std() > 50
    progress = torch.zeros(4096, dtype=torch.long) + 150
    hit = progress >= lim - 1
    assert 0.4 < hit.float().mean() < 0.6           # about half time out by step 150 instead of all at 299
