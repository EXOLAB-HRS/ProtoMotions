"""CPU tests of the sensory delay ring buffer (protomotions/envs/obs/sensory_delay.py)."""
from types import SimpleNamespace

import pytest
import torch

from protomotions.envs.obs.sensory_delay import SensoryDelayBuffer, SensoryDelayConfig, delay_actor_inputs

DT = 1.0 / 30.0
KEYS = dict(proprio_keys=["p"], vestib_keys=["v"])


def make(n=4, **kw):
    cfg = SensoryDelayConfig(enabled=True, **{**KEYS, **kw})
    return SensoryDelayBuffer(cfg, n, DT, "cpu", generator=torch.Generator().manual_seed(0))


def obs_at(t, n=4):
    """Obs value = step index t (+ env id * 1000 so envs are distinguishable)."""
    e = torch.arange(n, dtype=torch.float32).unsqueeze(-1) * 1000.0
    return {"p": (t + e).repeat(1, 3), "v": (t + e).repeat(1, 2), "cmd": torch.zeros(n, 1)}


def test_depth_and_ranges():
    b = make()
    assert b.depth == 5                     # 100 ms / 33.3 ms = 3 steps, + interpolation slot, + current
    b.reset(torch.arange(4), obs_at(0))
    assert ((b.delay_ms[:, 0] >= 40) & (b.delay_ms[:, 0] <= 70)).all()
    assert ((b.delay_ms[:, 1] >= 30) & (b.delay_ms[:, 1] <= 60)).all()
    with pytest.raises(ValueError):
        make(proprio_ms=(40.0, 120.0))
    with pytest.raises(KeyError):
        make().reset(torch.arange(4), {"p": torch.zeros(4, 3)})


def test_fixed_delay_interpolation():
    b = make(fixed_proprio_ms=55.0, fixed_vestib_ms=0.0)
    b.reset(torch.arange(4), obs_at(0))
    for t in range(1, 8):
        b.push(obs_at(t))
        out = b.read()
        assert set(out) == {"delayed_p", "delayed_v"}
        # 55 ms = 1.65 control steps: (1 - 0.65) * x(t - 1) + 0.65 * x(t - 2), clipped at the reset value 0
        want = max(t - 1.65, 0.0) if t >= 2 else 0.35 * (t - 1) + 0.65 * 0.0
        e = torch.arange(4, dtype=torch.float32) * 1000.0
        torch.testing.assert_close(out["delayed_p"][:, 0], want + e, atol=1e-3, rtol=0)
        torch.testing.assert_close(out["delayed_v"][:, 0], t + e)          # 0 ms = current


def test_exact_step_delays_and_max():
    for ms, steps in ((1000 * DT, 1), (2000 * DT, 2), (100.0, 3)):
        b = make(fixed_proprio_ms=ms, fixed_vestib_ms=0.0)
        b.reset(torch.arange(4), obs_at(0))
        for t in range(1, 10):
            b.push(obs_at(t))
        torch.testing.assert_close(b.read()["delayed_p"][:, 0], 9.0 - steps + torch.arange(4.0) * 1000, atol=1e-3, rtol=0)


def test_per_env_delays_vectorized():
    b = make()
    b.reset(torch.arange(4), obs_at(0))
    b.delay_ms[:, 0] = torch.tensor([40.0, 50.0, 60.0, 70.0])
    for t in range(1, 10):
        b.push(obs_at(t))
    got = b.read()["delayed_p"][:, 0] - torch.arange(4.0) * 1000
    torch.testing.assert_close(got, 9.0 - b.delay_ms[:, 0] / (1000 * DT), atol=1e-3, rtol=0)


def test_reset_fills_only_reset_envs():
    b = make(fixed_proprio_ms=70.0, fixed_vestib_ms=30.0)
    b.reset(torch.arange(4), obs_at(0))
    for t in range(1, 6):
        b.push(obs_at(t))
    before = b.read()
    o = obs_at(100)
    b.reset(torch.tensor([1, 3]), o)
    after = b.read()
    for k in after:
        torch.testing.assert_close(after[k][[0, 2]], before[k][[0, 2]])            # untouched envs
    torch.testing.assert_close(after["delayed_p"][[1, 3]], o["p"][[1, 3]])          # whole buffer = first obs
    b.push(obs_at(101))
    out = b.read()["delayed_p"][[1, 3], 0] - torch.tensor([1000.0, 3000.0])
    torch.testing.assert_close(out, torch.full((2,), 100.0))   # 70 ms still reads the reset value after one step


def test_resample_on_reset_and_state_roundtrip():
    b = make()
    b.reset(torch.arange(4), obs_at(0))
    d0 = b.delay_ms.clone()
    b.reset(torch.tensor([2]), obs_at(0))
    assert torch.equal(b.delay_ms[[0, 1, 3]], d0[[0, 1, 3]]) and not torch.equal(b.delay_ms[2], d0[2])
    for t in range(1, 4):
        b.push(obs_at(t))
    s = b.save_state()
    ref = b.read()
    b.push(obs_at(50))
    b.load_state(s)
    for k, v in b.read().items():
        torch.testing.assert_close(v, ref[k])


def test_disabled_and_actor_keys():
    mk = lambda keys: SimpleNamespace(in_keys=list(keys))
    base = ["max_coords_obs", "steering", "historical_max_coords_obs", "cadence_cond", "cadence_phase",
            "se_gravity", "se_specific_force", "terrain"]
    actor = mk(base)
    actor.mu_model = mk(base)
    m = SimpleNamespace(in_keys=base + ["disc_coords_obs"], actor=actor, critic=mk(base))
    cfg = SimpleNamespace(model=m)
    delay_actor_inputs(cfg, SensoryDelayConfig())                     # disabled: unchanged
    assert m.actor.in_keys == base
    delay_actor_inputs(cfg, SensoryDelayConfig(enabled=True))
    want = ["delayed_max_coords_obs", "steering", "delayed_historical_max_coords_obs", "cadence_cond",
            "cadence_phase", "delayed_se_gravity", "delayed_se_specific_force", "terrain"]
    assert m.actor.in_keys == want and m.actor.mu_model.in_keys == want
    assert m.critic.in_keys == base
    assert all(k in m.in_keys for k in want + base)
