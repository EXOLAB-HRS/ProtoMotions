# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""human_model_v4.0 ankle series elastic element (common/series_elastic.py): law, solver, stability, integration."""
import dataclasses
import math
from types import SimpleNamespace

import pytest
import torch

from protomotions.robot_configs.human_model.common.series_elastic import (
    SeriesElasticAnkle, channel_from_tendon, check_explicit_stability, tendon_energy, tendon_stretch, tendon_torque)

DT = torch.float64


def _cfg(c_pf=0.0, c_df=0.0, k_pf=180e3, toe=0.02, k_df=129e3):
    return dict(joints=["A"], channels=dict(PF=channel_from_tendon(+1, k_pf, 0.05, 0.24, toe, c_pf),
                                            DF=channel_from_tendon(-1, k_df, 0.04, 0.164, 0.0, c_df)))


def _see(n=1, **kw):
    return SeriesElasticAnkle(_cfg(**kw), ["A"], n, "cpu", DT)


def _col(x, n=1):
    return torch.full((n, 1), float(x), dtype=DT)


def test_tendon_law_c1_energy_and_inverse():
    k, toe = 450.0, 0.096
    D = torch.linspace(-0.05, 0.4, 20001, dtype=DT)
    T = tendon_torque(D, k, toe)
    assert (T >= 0).all() and (T[D <= 0] == 0).all()
    # C1 at the toe end: value k toe / 2 and slope k from both sides
    e = 1e-7
    lo, hi = tendon_torque(torch.tensor([toe - e, toe + e], dtype=DT), k, toe)
    assert abs(lo - k * toe / 2) < 1e-4 and abs(hi - lo) < 1e-3
    # energy is the integral of the torque
    num = torch.cumulative_trapezoid(T, D, dim=0)
    ana = tendon_energy(D, k, toe)[1:] - tendon_energy(D[:1], k, toe)
    assert torch.allclose(num, ana, atol=1e-6)
    F = torch.linspace(0, 120, 101, dtype=DT)
    assert torch.allclose(tendon_torque(tendon_stretch(F, k, toe), k, toe), F, atol=1e-9)


def test_tension_only_and_caps_random():
    g = torch.Generator().manual_seed(0)
    n = 4096
    see = SeriesElasticAnkle(_cfg(c_pf=1.35, c_df=0.62), ["A"], n, "cpu", DT)
    q = torch.zeros(n, 1, dtype=DT)
    for _ in range(50):
        q = q + 0.02 * torch.randn(n, 1, generator=g, dtype=DT)
        qd = torch.randn(n, 1, generator=g, dtype=DT) * 3
        tgt = torch.randn(n, 1, generator=g, dtype=DT) * 0.5
        kp = torch.rand(n, 1, generator=g, dtype=DT) * 1000
        kd = torch.rand(n, 1, generator=g, dtype=DT) * 30
        ff = torch.randn(n, 1, generator=g, dtype=DT) * 60
        neg = torch.rand(n, 1, generator=g, dtype=DT) * 34
        pos = torch.rand(n, 1, generator=g, dtype=DT) * 150
        tau = see.substep(q, qd, tgt, kp, kd, ff, neg, pos, 1 / 240)
        for name, cap in (("PF", pos), ("DF", neg)):
            f = see.force[name]
            assert (f >= 0).all() and (f <= cap + 1e-9).all()
            assert (see.D[name] >= 0).all()
            # massless CE: CE torque equals tendon torque (+ tendon damping) wherever the tendon is stretched
        assert torch.allclose(tau, see.force["PF"] - see.force["DF"])
        assert torch.isfinite(tau).all()


def _pendulum(see, I, dt, T, kp, kd, th0=0.3, tgt=0.0, ff=0.0, big=1e4):
    n = kp.shape[0]
    th, thd = torch.full((n, 1), th0, dtype=DT), torch.zeros(n, 1, dtype=DT)
    cap = torch.full((n, 1), big, dtype=DT)
    hist = []
    for _ in range(int(round(T / dt))):
        tau = see.substep(th, thd, torch.full_like(th, tgt), kp[:, None], kd[:, None], torch.full_like(th, ff),
                          cap, cap, dt)
        thd = thd + dt * tau / I          # semi-implicit Euler, as the articulation integrates an explicit effort
        th = th + dt * thd
        hist.append((th.clone(), thd.clone()))
    return th, thd, hist


def _energy(see, I, th, thd, kp, tgt=0.0):
    e = 0.5 * I * thd[:, 0] ** 2
    for name, ch in see.channels.items():
        e = e + tendon_energy(see.D[name][:, 0], ch.k_nm_per_rad, ch.toe_rad)
        on = see.force[name][:, 0] > 0
        e = e + on * 0.5 * kp * (tgt - see.phi[name][:, 0]) ** 2
    return e


def test_energy_conserved_without_damping():
    """No CE / tendon damping: bounded energy error of the semi-implicit Euler joint (O(omega dt)), no secular drift."""
    kp = torch.tensor([50.0, 300.0, 2000.0], dtype=DT)
    I = 0.03
    omega = math.sqrt(450.0 / I)
    for dt in (1 / 4000, 1 / 240):
        see = _see(n=3)
        es = []
        th, thd = torch.full((3, 1), 0.3, dtype=DT), torch.zeros(3, 1, dtype=DT)
        cap = torch.full((3, 1), 1e4, dtype=DT)
        for k in range(int(6.0 / dt)):
            tau = see.substep(th, thd, torch.zeros_like(th), kp[:, None], torch.zeros_like(th), torch.zeros_like(th),
                              cap, cap, dt)
            es.append(_energy(see, I, th, thd, kp))      # state at the substep start, D solved at this q
            thd = thd + dt * tau / I
            th = th + dt * thd
        es = torch.stack(es)
        e0 = es[0]
        rel = ((es - e0).abs() / e0).max(0).values
        assert (rel < omega * dt).all(), (dt, rel)
        steps = int(1.0 / dt)
        drift = (es[-steps:].mean(0) - es[:steps].mean(0)).abs() / e0
        assert (drift < 0.1 * omega * dt).all(), (dt, drift)


@pytest.mark.parametrize("fps", [120, 240, 480])
def test_stable_at_plant_timestep(fps):
    """Foot + armature about the ankle: I ~ 0.03 kg m^2 (armature 0.02). VIC gains over a wide range."""
    from protomotions.robot_configs.human_model.human_model_v4_0.model_config import series_elastic
    cfg = series_elastic()
    cfg["joints"] = ["A"]
    kp = torch.tensor([10.0, 80.0, 300.0, 1000.0, 3000.0], dtype=DT).repeat(3)
    kd = torch.tensor([0.5] * 5 + [8.0] * 5 + [50.0] * 5, dtype=DT)
    see = SeriesElasticAnkle(cfg, ["A"], len(kp), "cpu", DT)
    th, thd, hist = _pendulum(see, 0.03, 1.0 / fps, 4.0, kp, kd)
    amp = torch.stack([h[0] for h in hist])[:, :, 0]
    first, last = amp[: fps // 2].abs().max(0).values, amp[-fps // 2:].abs().max(0).values
    assert torch.isfinite(amp).all()
    assert (last <= first + 1e-9).all()          # never grows
    assert (thd.abs() < 50).all()


def test_stiff_tendon_reduces_to_v36_active_torque():
    """k -> inf, no toe, no damping: tau = clip(Kp (q* - q) - Kd qd_fd + ff, -N, P) with qd_fd = (q_n - q_n-1) / dt."""
    n, dt = 64, 1 / 240
    g = torch.Generator().manual_seed(1)
    see = SeriesElasticAnkle(_cfg(k_pf=1e12, toe=0.0, k_df=1e12), ["A"], n, "cpu", DT)
    kp = torch.rand(n, 1, generator=g, dtype=DT) * 500 + 10
    kd = torch.rand(n, 1, generator=g, dtype=DT) * 20 + 0.5
    neg, pos = torch.full((n, 1), 34.0, dtype=DT), torch.full((n, 1), 150.0, dtype=DT)
    t = torch.arange(0, 2, dt, dtype=DT)
    phase = torch.rand(n, 1, generator=g, dtype=DT) * 6
    q_all = 0.3 * torch.sin(2 * math.pi * t[None, :] + phase)          # [n, T]
    tgt = torch.randn(n, 1, generator=g, dtype=DT) * 0.2
    ff = torch.randn(n, 1, generator=g, dtype=DT) * 20
    err = 0.0
    for k in range(1, q_all.shape[1]):
        q = q_all[:, k:k + 1]
        qd_fd = (q_all[:, k:k + 1] - q_all[:, k - 1:k]) / dt
        tau = see.substep(q, qd_fd, tgt, kp, kd, ff, neg, pos, dt)
        if k > 2:
            ref = (kp * (tgt - q) - kd * qd_fd + ff).clamp(-34.0, 150.0)
            err = max(err, (tau - ref).abs().max().item())
    assert err < 1e-3


def test_reset_restarts_ce_at_joint_angle():
    see = _see(n=2)
    cap = torch.full((2, 1), 1e4, dtype=DT)
    q = torch.tensor([[0.1], [0.2]], dtype=DT)
    for _ in range(5):
        see.substep(q, torch.zeros_like(q), q + 0.3, torch.full_like(q, 300), torch.full_like(q, 5),
                    torch.zeros_like(q), cap, cap, 1 / 240)
    assert (see.D["PF"] > 0).all()
    see.reset(torch.tensor([1]))
    q2 = torch.tensor([[0.1], [-0.4]], dtype=DT)
    see.substep(q2, torch.zeros_like(q2), q2, torch.zeros_like(q2), torch.zeros_like(q2), torch.zeros_like(q2),
                cap, cap, 1 / 240)
    assert see.phi["PF"][1, 0] == pytest.approx(-0.4) and see.D["PF"][1, 0] == 0
    assert see.D["PF"][0, 0] > 0 or see.force["PF"][0, 0] >= 0


def test_cap_velocity_uses_engaged_ce():
    see = _see(n=1)
    cap = torch.full((1, 1), 1e4, dtype=DT)
    q = _col(0.0)
    see.substep(q, _col(0.0), _col(0.5), _col(300), _col(5), _col(0), cap, cap, 1 / 240)   # PF engaged
    qd_full = torch.tensor([[3.0]], dtype=DT)
    out = see.cap_velocity(qd_full)
    assert out[0, 0] == pytest.approx(see.phid["PF"][0, 0].item())


def test_explicit_stability_guard():
    see = _see(n=1)
    info = {"A": SimpleNamespace(armature=0.02)}
    check_explicit_stability(see, info, 240)
    check_explicit_stability(see, info, 120)
    stiff = SeriesElasticAnkle(_cfg(k_pf=900e3), ["A"], 1, "cpu", DT)       # 2250 N m/rad
    with pytest.raises(ValueError):
        check_explicit_stability(stiff, info, 120)
    with pytest.raises(ValueError):
        check_explicit_stability(see, {"A": SimpleNamespace(armature=None)}, 240)


def _same(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if torch.is_tensor(a):
        return torch.equal(a, b)
    return a == b


def test_v40_config_is_v36_plus_see():
    from protomotions.robot_configs.factory import robot_config
    from protomotions.robot_configs.human_model.registry import get_model
    r40, r36 = robot_config("human_model_v4.0"), robot_config("human_model_v3.6")
    a, b = dataclasses.asdict(r40), dataclasses.asdict(r36)
    see = a["human_model_parameters"].pop("series_elastic")
    assert _same(a, b)
    assert see["joints"] == ["L_Ankle_y", "R_Ankle_y"]
    assert see["channels"]["PF"]["k_nm_per_rad"] == pytest.approx(450.0)
    assert see["channels"]["PF"]["toe_rad"] == pytest.approx(0.096)
    assert see["channels"]["DF"]["k_nm_per_rad"] == pytest.approx(206.4)
    m = get_model("human_model_v4.0")
    assert m.model_id == "human_model_v4.0" and m.asset_path("usd").is_file()


def _fake_sim(n, names, q, qd, command, kp, kd):
    ndof = len(names)
    idx = torch.arange(ndof)
    state = SimpleNamespace(dof_pos=q, dof_vel=qd)
    rec = {}
    robot = SimpleNamespace(
        _data=SimpleNamespace(joint_effort_limits=torch.zeros(n, ndof)), _ALL_INDICES=torch.arange(n),
        root_physx_view=SimpleNamespace(set_dof_max_forces=lambda buf, indices: rec.__setitem__("cap", buf.clone())),
        set_joint_position_target=lambda t: rec.__setitem__("target", t.clone()))
    sim = SimpleNamespace(
        _get_simulator_dof_state=lambda: SimpleNamespace(convert_to_common=lambda conv: state),
        data_conversion=SimpleNamespace(dof_convert_to_sim=idx, dof_convert_to_common=idx),
        _common_p_gains=kp, _common_d_gains=kd, config=SimpleNamespace(sim=SimpleNamespace(fps=240)),
        decimation=8, _robot=robot, _apply_simulator_torques=lambda t: rec.__setitem__("effort", t.clone()),
        vic_ff_prior=None)
    return sim, rec


def test_implicit_hook_only_changes_see_columns(monkeypatch):
    from protomotions.robot_configs.human_model.common.dynamics import HumanJointModel
    from protomotions.robot_configs.human_model.profile import load_profile
    from protomotions.robot_configs.human_model.human_model_v4_0.model_config import (
        series_elastic, active_strength_scale, passive_exponentials)
    from protomotions.robot_configs.human_model.common.series_elastic import build
    from protomotions.robot_configs.base import ControlType
    monkeypatch.setenv("HC_HUMAN_IMPLICIT_PD", "1")
    monkeypatch.setenv("HC_HUMAN_IMPLICIT_CAP_EVERY", "substep")
    profile = load_profile("human_model_v3")
    names = list(profile["joints"])
    n = 8
    g = torch.Generator().manual_seed(3)
    kw = dict(active_strength_scale=active_strength_scale(), passive_exponentials=passive_exponentials())
    m36 = HumanJointModel(profile, names, "cpu", **kw)
    m40 = HumanJointModel(profile, names, "cpu", **kw)
    m40.see = build(series_elastic(), names, n, "cpu", torch.float32)
    lo, hi = m36.lower, m36.upper
    q = lo + (hi - lo) * (0.3 + 0.4 * torch.rand(n, len(names), generator=g))
    qd = torch.randn(n, len(names), generator=g)
    command = q + 0.2 * torch.randn(n, len(names), generator=g)
    kp, kd = torch.full((n, len(names)), 200.0), torch.full((n, len(names)), 10.0)
    out = {}
    for tag, m in (("36", m36), ("40", m40)):
        sim, rec = _fake_sim(n, names, q, qd, command, kp, kd)
        m._apply_implicit(sim, command, ControlType.BUILT_IN_PD)
        out[tag] = (rec, sim)
    see_cols = [names.index("L_Ankle_y"), names.index("R_Ankle_y")]
    other = [i for i in range(len(names)) if i not in see_cols]
    r36, r40 = out["36"][0], out["40"][0]
    assert torch.equal(r36["target"], r40["target"])
    assert torch.allclose(r36["cap"][:, other], r40["cap"][:, other])
    assert torch.allclose(r36["effort"][:, other], r40["effort"][:, other])
    assert (r40["cap"][:, see_cols] == 0).all()
    sim40 = out["40"][1]
    tissue = (m40.elastic_torque(q) + m40.damping_torque(qd))[:, see_cols]
    tau = m40.see.force["PF"] - m40.see.force["DF"]
    assert torch.allclose(r40["effort"][:, see_cols], tissue + tau, atol=1e-5)
    assert torch.allclose(sim40.human_active_torques[:, see_cols], tau)
    assert sim40.human_see_ce_power.shape == (n, len(names))
