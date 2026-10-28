"""CPU tests: vic_tc_v3 stiffness law (vic_steering.py) and the symmetry dead zone (gait_symmetry.py), 2026-10-28."""
import math
from types import SimpleNamespace

import pytest
import torch

from protomotions.envs.control import vic_steering as vs
from protomotions.envs.control import gait_symmetry as gs

JOINTS = ("L_Hip_y", "L_Knee_y", "L_Ankle_y", "R_Hip_y", "R_Knee_y", "R_Ankle_y",
          "L_Hip_x", "R_Hip_x", "L_Subtalar_x", "R_Subtalar_x")


def test_defaults_are_backward_compatible():
    c = vs.VicSteeringConfig()
    assert c.vic_law == "v2"
    s = gs.VicSymSteeringConfig()
    assert s.sym_deadzone is False and s.sym_stance_tol_pct == 0.0


def test_vic3_defaults():
    p = vs.vic3_defaults("Ankle_y")
    assert p["k_base"] == 22.0 and p["c"] == 3.5
    assert p["k_co_max"] == pytest.approx(406 - 22)
    # d = geometric mean over states of b_centre / sqrt(k_centre)
    r = [(22, 100, 0.6, 1.5), (22, 406, 0.6, 1.5), (290, 340, 0.6, 1.5)]
    d = math.exp(sum(math.log(math.sqrt(bl * bh) / math.sqrt(math.sqrt(lo * hi))) for lo, hi, bl, bh in r) / 3)
    assert p["d"] == pytest.approx(d)
    assert vs.vic3_defaults("Knee_y", {"Knee_y": {"c": 1.0}})["c"] == 1.0
    with pytest.raises(ValueError):
        vs.vic3_defaults("Toe_y")


def test_vic3_stiffness_terms():
    kb, c, kco = torch.tensor([20.0]), torch.tensor([3.0]), torch.tensor([300.0])
    tau = torch.tensor([[0.0], [10.0], [40.0]])
    u = torch.tensor([[0.0], [0.0], [0.5]])
    b, l, co = vs.vic3_stiffness(kb, c, tau, kco, u)
    assert torch.equal(b, torch.full((3, 1), 20.0))
    assert torch.allclose(l, torch.tensor([[0.0], [30.0], [120.0]]))
    assert torch.allclose(co, torch.tensor([[0.0], [0.0], [150.0]]))
    _, l2, _ = vs.vic3_stiffness(kb, c, tau, kco, u, qd=torch.full((3, 1), 2.0), srs_omega=2.0)
    assert torch.allclose(l2, 0.5 * l)                                   # omega / (omega + |qd|) = 0.5


def test_guard_dead_zone():
    lo, hi = torch.log(torch.tensor([[10.0]])), torch.log(torch.tensor([[100.0]]))
    inside = torch.log(torch.tensor([[11.0], [50.0], [99.0]])).requires_grad_(True)
    g = vs.vic3_guard(inside, lo.expand(3, 1), hi.expand(3, 1))
    assert torch.equal(g, torch.zeros(3))
    g.sum().backward()
    assert torch.equal(inside.grad, torch.zeros(3, 1))                   # no pull toward the centre
    out = vs.vic3_guard(torch.log(torch.tensor([[200.0], [5.0]])), lo.expand(2, 1), hi.expand(2, 1))
    assert torch.allclose(out, torch.tensor([1.0, 1.0]))                 # one octave outside each side


def test_symmetry_error_default_identical_and_dead_zone():
    x, ref, sig = torch.randn(64, 7) * 10, torch.randn(64, 7) * 10, torch.full((7,), 5.0)
    assert torch.equal(gs.symmetry_error(x, ref, sig), (x - ref) / sig)  # default path bit-identical
    tol = torch.tensor([gs.SYM_TOL_DEG[j] for j, _, _ in gs.VicSymSteeringConfig.sym_joints])
    e = gs.symmetry_error(ref + 0.99 * tol, ref, sig, tol)
    assert torch.allclose(e, torch.zeros_like(e), atol=1e-5)
    e = gs.symmetry_error(ref - tol - 5.0, ref, sig, tol)
    assert torch.allclose(e, torch.ones_like(e), atol=1e-5)             # 5 deg beyond the tolerance = 1 sigma


def test_stance_symmetry_factor():
    frac = torch.tensor([[60.0, 57.0], [62.0, 54.27], [60.0, 50.0]])
    ok = torch.tensor([True, True, False])
    f = gs.stance_symmetry_factor(frac, ok, 3.73, 2.0)
    assert f[0] == 1.0 and f[2] == 1.0
    assert f[1] == pytest.approx(math.exp(-0.5 * ((7.73 - 3.73) / 2.0) ** 2), rel=1e-4)


# ---------------------------------------------------------------- v3 control step on a stub simulator
class _Dof:
    def __init__(self, q, qd):
        self.dof_pos, self.dof_vel = q, qd

    def convert_to_common(self, _):
        return self


def _stub(n=4, law_kwargs=None):
    J = len(JOINTS)
    ndof = J + 2
    cfg = vs.VicSteeringConfig(vic_joints=JOINTS, vic_law="v3", delta_tau_up_s=0.05, delta_tau_down_s=0.15,
                               **(law_kwargs or {}))
    sim = SimpleNamespace(_common_p_gains=torch.zeros(n, ndof), _common_d_gains=torch.zeros(n, ndof),
                          vic_ff_prior=torch.zeros(n, ndof), vic_ff_res_target=torch.zeros(n, ndof),
                          vic_ff_res=torch.zeros(n, ndof), data_conversion=None,
                          human_active_torques=torch.zeros(n, ndof))
    sim.q, sim.qd = torch.zeros(n, ndof), torch.zeros(n, ndof)
    sim._get_simulator_dof_state = lambda: _Dof(sim.q, sim.qd)
    env = SimpleNamespace(simulator=sim, dt=1 / 30, progress_buf=torch.full((n,), 10))
    self = object.__new__(vs.VicSteering)
    self.config, self.env = cfg, env
    self._idx = torch.arange(J)
    self._delta = torch.zeros(n, J)
    self._k_old = torch.full((J,), 100.0)
    self._b_old = torch.full((J,), 10.0)
    self._tau_max = torch.full((J,), 150.0)
    self.alpha, self.beta = 1.0, 0.0
    self._gain_log_path = None
    self._write_gains = lambda: None
    self._log = lambda: None
    self._sag = torch.arange(6)
    self._bw = 63.3 * 9.81
    self._mean_k = torch.zeros(3, J)
    self._init_v3(n, torch.device("cpu"))
    return self, sim


def _step(self, sim, a, state, fresh=False):
    n, J = state.shape
    jj = torch.arange(J).expand_as(state)
    s = 1.0 - math.exp(-self.env.dt / self.config.gain_smooth_s)
    self._pre_physics_v3(None, torch.zeros(n, J + 2), torch.tanh(a), torch.zeros(n, J), state, jj,
                         torch.full((n,), fresh), s)
    return sim._common_p_gains[:, :J].clone(), sim._common_d_gains[:, :J].clone()


def test_v3_step_relaxed_load_and_cocontraction():
    self, sim = _stub()
    n, J = 4, len(JOINTS)
    state = torch.ones(n, J, dtype=torch.long)                           # stance
    kp, kd = _step(self, sim, torch.full((n, J), -2.0), state, fresh=True)
    kb = self._v3_kb
    lo = torch.tensor([vs.vic3_defaults(j[2:])["k"][1][0] for j in JOINTS])
    assert torch.allclose(kp[0], torch.maximum(kb, 0.3 * lo))            # relaxed, no load, corridor floor
    assert torch.allclose(kd[0], (self._v3_d * kp[0].sqrt()).clamp(
        min=torch.tensor([0.3 * vs.vic3_defaults(j[2:])["b"][1][0] for j in JOINTS])), rtol=1e-5)
    # Constant 40 N m on every joint: the load term rises with the 50 ms filter toward c * 40.
    sim.human_active_torques[:] = 40.0
    for _ in range(60):
        kp, _ = _step(self, sim, torch.full((n, J), -2.0), state)
    assert torch.allclose(self._v3_terms[1, 0], self._v3_c * 40.0, rtol=1e-3)
    assert torch.equal(self._v3_terms[2], torch.zeros(n, J))             # u = relu(tanh(-2)) = 0
    # u -> 1 adds k_co_max (filtered), cost = k_co / (c tau_max) mean.
    for _ in range(90):
        kp, _ = _step(self, sim, torch.full((n, J), 20.0), state)
    assert torch.allclose(self._v3_terms[2, 0], self._v3_kco, rtol=1e-3)
    want = (self._v3_kco / (self._v3_c * self._tau_max)).mean()
    assert self._cocon[0] == pytest.approx(float(want), rel=1e-3)
    hi = torch.tensor([2.0 * vs.vic3_defaults(j[2:])["k"][1][1] for j in JOINTS])
    assert (kp[0] <= hi * (1 + 1e-5)).all()


def test_v3_guard_zero_inside_and_standing_needs_u():
    self, sim = _stub()
    n, J = 4, len(JOINTS)
    stand = torch.full((n, J), 2, dtype=torch.long)
    sim.human_active_torques[:] = 8.0                                    # quiet standing ankle torque scale
    for _ in range(60):
        kp, _ = _step(self, sim, torch.full((n, J), -2.0), stand)
    ank = JOINTS.index("L_Ankle_y")
    assert kp[0, ank] < 290 / 1.25 and self._band[0] > 0                 # load alone: below the standing range
    for _ in range(90):
        kp, _ = _step(self, sim, torch.full((n, J), 0.62), stand)        # u = tanh(0.62) ~ 0.55
    assert 290 / 1.25 <= kp[0, ank] <= 340 * 1.25


def test_v3_alpha_zero_keeps_parent_gains():
    self, sim = _stub()
    self.alpha = 0.0
    n, J = 4, len(JOINTS)
    kp, kd = _step(self, sim, torch.zeros(n, J), torch.zeros(n, J, dtype=torch.long), fresh=True)
    assert torch.allclose(kp, torch.full((n, J), 100.0)) and torch.allclose(kd, torch.full((n, J), 10.0))
