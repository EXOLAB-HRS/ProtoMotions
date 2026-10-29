"""CPU tests of the joint stretch reflex (protomotions/robot_configs/human_model/common/joint_reflex.py)."""
from types import SimpleNamespace

import pytest
import torch

from protomotions.robot_configs.human_model.common.joint_reflex import (
    JointReflex, JointReflexConfig, build_joint_reflex)
from protomotions.robot_configs.human_model.common.dynamics import _vic_feedforward

HZ = 240.0
DT_CTRL = 1.0 / 30.0
DOFS = ["A_y", "L_Knee_y", "B_x", "R_Knee_y"]
JOINTS = ("L_Knee_y", "R_Knee_y")


def make(n=3, **kw):
    kw = {"enabled": True, "joints": JOINTS, "fixed_delay_ms": 1000.0 * 8 / HZ, "v_th": {"Knee_y": 1.0},
          "k_v": {"Knee_y": 2.0}, **kw}
    cfg = JointReflexConfig(**kw)
    return JointReflex(cfg, DOFS, {j: 100.0 for j in JOINTS}, physics_hz=HZ, control_dt=DT_CTRL, num_envs=n,
                       device="cpu", generator=torch.Generator().manual_seed(0))


def start(r, q_ref=None):
    """First control step after reset: qdot_ref taken from qdot (= 0 here)."""
    q_ref = torch.zeros(r.n, len(DOFS)) if q_ref is None else q_ref
    r.control_step(q_ref, torch.zeros(r.n, len(DOFS)))


def test_delay_buffer_exact_steps():
    r = make()
    assert r.delay_steps.tolist() == [8, 8, 8]
    start(r)
    out = []
    for t in range(20):
        qd = torch.zeros(r.n, 2)
        q = torch.zeros(r.n, 2)
        if t == 3:                                   # one-substep stretch spike on env 1, left knee
            qd[1, 0], q[1, 0] = 5.0, 0.1
        out.append(r.compute(q, qd).clone())
    tau = torch.stack(out)                            # [T, N, J]
    nz = (tau != 0).nonzero().tolist()
    assert nz == [[3 + 8, 1, 0]]                      # arrives exactly d = 8 substeps (33.3 ms) later
    assert tau[11, 1, 0].item() == pytest.approx(-2.0 * (5.0 - 1.0))


def test_threshold_silence_and_active_fraction():
    r = make()
    start(r)
    for _ in range(30):
        tau = r.compute(torch.full((r.n, 2), 0.05), torch.full((r.n, 2), 0.99))   # below v_th, deviation growing
        assert torch.all(tau == 0)
    r.control_step(torch.zeros(r.n, len(DOFS)))
    a, f = r.step_stats()
    assert torch.all(a == 0) and torch.all(f == 0)
    logs = r.log_values()
    assert logs["joint_reflex/active_L_Knee_y"].item() == 0.0
    assert logs["joint_reflex/delay_ms"].item() == pytest.approx(1000.0 * 8 / HZ)


def test_direction_and_away_gate():
    r = make(fixed_delay_ms=0.0)
    start(r)
    q = torch.tensor([[0.1, -0.1], [-0.1, 0.1], [0.1, 0.0]])
    qd = torch.tensor([[3.0, -3.0], [3.0, -3.0], [0.0, 0.0]])
    tau = r.compute(q, qd)
    # env 0: deviation grows (positive and negative group) -> torque opposes the velocity, 2 * (3 - 1) = 4
    assert tau[0].tolist() == pytest.approx([-4.0, 4.0])
    # env 1: moving back toward q_ref -> silent with away_only
    assert tau[1].tolist() == [0.0, 0.0]
    r2 = make(fixed_delay_ms=0.0, away_only=False)
    start(r2)
    assert r2.compute(q, qd)[1].tolist() == pytest.approx([-4.0, 4.0])


def test_deviation_velocity_relative_to_reference():
    r = make(fixed_delay_ms=0.0)
    start(r)
    q_ref = torch.zeros(r.n, len(DOFS))
    q_ref[:, 1] = 3.0 * DT_CTRL                        # target of the left knee moves at 3 rad/s
    r.control_step(q_ref)
    assert r.qd_ref[:, 0].tolist() == pytest.approx([3.0] * r.n)
    q = torch.full((r.n, 2), 0.2)
    q[:, 0] = q_ref[:, 1] + 0.05
    qd = torch.zeros(r.n, 2)
    qd[:, 0] = 3.5                                     # following the reference motion: e_v = 0.5 < v_th
    assert torch.all(r.compute(q, qd)[:, 0] == 0)
    qd[:, 0] = 6.0                                     # pushed: e_v = 3 -> -2 * (3 - 1)
    assert r.compute(q, qd)[:, 0].tolist() == pytest.approx([-4.0] * r.n)


def test_gain_modulation_by_coactivation():
    r = make(fixed_delay_ms=0.0, kappa=1.0)
    vic = SimpleNamespace(config=SimpleNamespace(vic_joints=("L_Hip_y", "L_Knee_y", "R_Knee_y")),
                          coactivation_level=lambda: torch.tensor([[0.0, 0.5, -0.8]] * 3))
    r.attach_vic(vic)
    start(r)
    tau = r.compute(torch.full((r.n, 2), 0.1), torch.full((r.n, 2), 3.0))
    assert tau[:, 0].tolist() == pytest.approx([-4.0 * 1.5] * r.n)    # u = 0.5 -> x1.5
    assert tau[:, 1].tolist() == pytest.approx([-4.0] * r.n)          # u < 0 -> g0
    # Fallback without a getter: filtered delta / bound, as kept by vic_steering.
    vic2 = SimpleNamespace(config=vic.config, _delta=torch.tensor([[0.0, 1.0, 0.0]] * 3),
                           _dmax=torch.tensor([[1.0, 2.0, 1.0]] * 3))
    r.attach_vic(vic2)
    r.control_step(torch.zeros(r.n, len(DOFS)))
    assert r.u[:, 0].tolist() == pytest.approx([0.5] * r.n)


def test_clip_and_tonic():
    r = make(fixed_delay_ms=0.0, max_frac=0.05)
    start(r)
    assert r.compute(torch.full((r.n, 2), 0.1), torch.full((r.n, 2), 50.0))[0, 0].item() == pytest.approx(-5.0)
    r = make(fixed_delay_ms=0.0, k_l={"Knee_y": 10.0}, l_th={"Knee_y": 0.2})
    start(r)
    tau = r.compute(torch.full((r.n, 2), 0.3), torch.zeros(r.n, 2))       # length only: 10 * (0.3 - 0.2)
    assert tau[0, 0].item() == pytest.approx(-1.0)


def test_reset_zero_prehistory_and_new_delay():
    r = make(fixed_delay_ms=None, delay_ms=(30.0, 40.0))
    assert ((r.delay_steps >= 7) & (r.delay_steps <= 10)).all()           # 30-40 ms at 240 Hz
    start(r)
    for _ in range(12):
        r.compute(torch.full((r.n, 2), 0.1), torch.full((r.n, 2), 4.0))
    assert (r.last_torque() != 0).all()
    r.reset(torch.tensor([1]))
    assert torch.all(r.history[:, 1] == 0) and bool(r.fresh[1]) and not bool(r.fresh[0])
    qd = torch.zeros(r.n, len(DOFS))
    qd[:, 1] = 2.0
    r.control_step(torch.zeros(r.n, len(DOFS)), qd)
    assert r.qd_ref[1, 0].item() == pytest.approx(2.0)                     # first step: qdot_ref = qdot
    d1 = int(r.delay_steps[1])
    for t in range(d1):
        tau = r.compute(torch.full((r.n, 2), 0.1), torch.full((r.n, 2), 6.0))
        if t < d1 - 1:
            assert torch.all(tau[1] == 0)                                  # silent until the delay has passed
    assert (r.compute(torch.full((r.n, 2), 0.1), torch.full((r.n, 2), 6.0))[1] != 0).all()


def test_save_load_state():
    r = make()
    start(r)
    r.compute(torch.full((r.n, 2), 0.1), torch.full((r.n, 2), 4.0))
    s = r.save_state()
    a = [r.compute(torch.full((r.n, 2), 0.1), torch.full((r.n, 2), 4.0)).clone() for _ in range(10)]
    r.load_state(s)
    b = [r.compute(torch.full((r.n, 2), 0.1), torch.full((r.n, 2), 4.0)).clone() for _ in range(10)]
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_calibration_histogram():
    r = make(fixed_delay_ms=0.0, calibrate=True, gain_scale=0.0)
    start(r)
    for k in range(100):
        tau = r.compute(torch.full((r.n, 2), 0.1), torch.full((r.n, 2), 0.01 * k))
        assert torch.all(tau == 0)                                       # sensor only
    q = r.calibration_quantiles()
    assert q["all_samples"] == 300
    assert q["all"]["L_Knee_y"]["0.5"] == pytest.approx(0.5, abs=0.02)
    assert q["all"]["L_Knee_y"]["0.99"] == pytest.approx(0.99, abs=0.02)


def test_default_off_and_feedforward_unchanged():
    from protomotions.envs.base_env.config import EnvConfig
    assert EnvConfig.__dataclass_fields__["joint_reflex"].default is None
    assert not JointReflexConfig().enabled
    assert build_joint_reflex(None, None) is None and build_joint_reflex(JointReflexConfig(), None) is None
    g = torch.Generator().manual_seed(1)
    args = [torch.randn(4, 5, generator=g) * 20 for _ in range(3)]
    neg, pos = torch.rand(4, 5, generator=g) * 30, torch.rand(4, 5, generator=g) * 30
    req = torch.randn(4, 5, generator=g) * 30
    res_a, res_b = args[2].clone(), args[2].clone()
    a = _vic_feedforward(args[0], args[1], res_a, torch.tensor(0.1), req, neg, pos)
    b = _vic_feedforward(args[0], args[1], res_b, torch.tensor(0.1), req, neg, pos, None)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    res_c = args[2].clone()
    ff_c, _, _ = _vic_feedforward(args[0], args[1], res_c, torch.tensor(0.1), req, neg, pos, torch.full((4, 5), 1e3))
    assert torch.equal(ff_c, pos)                                         # reflex + ff clipped to the strength cap


def test_substep_writes_reflex_buffer(monkeypatch):
    r = make(fixed_delay_ms=0.0)
    start(r)
    q = torch.zeros(r.n, len(DOFS))
    qd = torch.zeros(r.n, len(DOFS))
    q[:, 1], qd[:, 1] = 0.1, 3.0                       # L_Knee_y stretched
    dof = SimpleNamespace(dof_pos=q, dof_vel=qd)
    sim = SimpleNamespace(vic_ff_prior=torch.zeros(r.n, len(DOFS)), data_conversion=None,
                          _get_simulator_dof_state=lambda: SimpleNamespace(convert_to_common=lambda conv: dof))
    monkeypatch.setenv("HC_HUMAN_IMPLICIT_PD", "0")
    with pytest.raises(RuntimeError):
        r.substep(sim)
    monkeypatch.setenv("HC_HUMAN_IMPLICIT_PD", "1")
    r.substep(sim)
    buf = sim.vic_ff_reflex
    assert buf[:, 1].tolist() == pytest.approx([-4.0] * r.n)
    assert torch.all(buf[:, [0, 2, 3]] == 0)
    ptr = buf.data_ptr()
    r.substep(sim)
    assert sim.vic_ff_reflex.data_ptr() == ptr          # in place: a static input of the captured CUDA graph


def test_highpass_signal_silent_on_smooth_motion_fires_on_jump():
    """signal="highpass": a smooth 1 Hz swing (1.5 rad/s peak; high-pass gain 0.53 at 1 Hz) stays below v_th; a sudden 3 rad/s velocity jump fires."""
    import math
    import torch
    from protomotions.robot_configs.human_model.common.joint_reflex import JointReflex, JointReflexConfig
    cfg = JointReflexConfig(enabled=True, joints=("L_Knee_y",), delay_ms=(0.0, 0.0), v_th={"Knee_y": 1.0},
                            k_v={"Knee_y": 2.0}, signal="highpass", hp_tau_s=0.1, max_frac=1.0)
    r = JointReflex(cfg, ["L_Knee_y"], {"L_Knee_y": 100.0}, physics_hz=240, control_dt=1 / 30, num_envs=1, device="cpu")
    r.reset(torch.tensor([0]))
    peak = 0.0
    for k in range(480):                                     # 2 s smooth motion
        t = k / 240
        qd = torch.tensor([[1.5 * math.cos(2 * math.pi * t)]])
        tau = r.compute(torch.zeros(1, 1), qd)
        if k > 48:
            peak = max(peak, float(tau.abs().max()))
    assert peak == 0.0, peak
    fired = 0.0
    for k in range(24):                                      # sudden +3 rad/s jump on top
        qd = torch.tensor([[1.5 + 3.0]])
        fired = max(fired, float(tau.abs().max()) if k == 0 else float(r.compute(torch.zeros(1, 1), qd).abs().max()))
    assert fired > 0.0
