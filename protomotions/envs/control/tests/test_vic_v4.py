"""CPU tests: vic_tc_v4 impedance prior helpers (vic_steering.py), 2026-10-30."""
import math

import pytest
import torch

from protomotions.envs.control import vic_steering as vs


def test_v4_is_opt_in():
    c = vs.VicSteeringConfig()
    assert c.vic_law == "v2" and c.v4_band_margin == 2.0 and c.v4_passive is True


def test_leg_weights_continuous_and_sum_to_one():
    load = torch.tensor([[0.0, 0.0], [0.125, 0.6], [0.6, 0.6], [0.6, 0.6]])
    slow = torch.tensor([1.0, 1.0, 0.0, 1.0])
    w = vs.vic4_leg_weights(load, slow, 0.05, 0.20)
    assert torch.allclose(w.sum(-1), torch.ones(4, 2))
    assert torch.allclose(w[0, 0], torch.tensor([1.0, 0.0, 0.0]))            # airborne: swing
    assert torch.allclose(w[1, 0], torch.tensor([0.25, 0.25, 0.5]))        # half-loaded leg, standing factor 0.5
    assert torch.allclose(w[2, 0], torch.tensor([0.0, 1.0, 0.0]))            # walking double support: stance
    assert torch.allclose(w[3, 1], torch.tensor([0.0, 0.0, 1.0]))            # slow, both loaded: stand
    # No jump: a small change of load gives a small change of weights.
    w1 = vs.vic4_leg_weights(load[1:2] + 1e-3, slow[1:2], 0.05, 0.20)
    assert (w1 - w[1:2]).abs().max() < 0.05


def test_damping_law_d1():
    lk_c, lb_c = torch.tensor([[math.log(100.0)]]), torch.tensor([[math.log(1.0)]])
    lo, hi = torch.tensor([[math.log(0.6)]]), torch.tensor([[math.log(1.5)]])
    a = torch.tensor([0.15])
    lb = vs.vic4_damping(lk_c + math.log(4.0), lk_c, lb_c, a, lo, hi)
    assert lb.exp().item() == pytest.approx(4.0 ** 0.15)                    # ankle: b barely follows k
    lb2 = vs.vic4_damping(lk_c + 10.0, lk_c, lb_c, torch.tensor([0.5]), lo, hi)
    assert lb2.exp().item() == pytest.approx(3.0)                            # clamped at 2 b_hi


def test_coactivation_beyond_load_stiffness():
    k_act, c = torch.tensor([[30.0, 300.0]]), torch.tensor([3.0, 3.0])
    tau, tau_max = torch.tensor([[20.0, 20.0]]), torch.tensor([100.0, 100.0])
    tau_co, u = vs.vic4_coactivation(k_act, c, tau, tau_max)
    assert tau_co[0, 0] == 0.0                                               # load stiffness already 60 > 30
    assert tau_co[0, 1] == pytest.approx(80.0) and u[0, 1] == pytest.approx(0.8)


def test_dead_zone_and_prior():
    lk_c, h = torch.tensor([[math.log(100.0)]]), torch.tensor([[math.log(2.0)]])
    inside = torch.log(torch.tensor([[60.0], [190.0]])).reshape(2, 1)
    assert torch.equal(vs.vic4_dead_zone(inside, lk_c, h), torch.zeros(2, 1))
    out = vs.vic4_dead_zone(torch.log(torch.tensor([[400.0]])), lk_c, h)
    assert out.item() == pytest.approx(math.log(2.0))


def test_passive_diag_matches_plant_tangent():
    from protomotions.robot_configs.human_model.common.profile import load_profile
    from protomotions.robot_configs.human_model.common.dynamics import HumanJointModel
    p = load_profile("human_model_v3")
    names = list(p["joints"])
    m = HumanJointModel(p, names)
    idx = torch.tensor([names.index(j) for j in ("L_Hip_y", "L_Knee_y", "L_Ankle_y")])
    q = (torch.rand(5, len(names)) - 0.5) * 0.6
    k, b = vs.vic4_passive_diag(m, q, idx)
    full = m.passive_impedance(q, torch.zeros_like(q))["stiffness_nm_per_rad"]
    assert torch.allclose(k, torch.diagonal(full, dim1=-2, dim2=-1)[:, idx], rtol=1e-4, atol=1e-4)
    assert torch.equal(b, m.damping[idx])


# ---------------------------------------------------------------- stepped stub (no simulator)
from types import SimpleNamespace  # noqa: E402

JOINTS = ("L_Hip_y", "L_Knee_y", "L_Ankle_y", "R_Hip_y", "R_Knee_y", "R_Ankle_y",
          "L_Hip_x", "R_Hip_x", "L_Subtalar_x", "R_Subtalar_x")


class _Dof:
    def __init__(self, q, qd):
        self.dof_pos, self.dof_vel = q, qd

    def convert_to_common(self, _):
        return self


def _stub(n=4, k_pass=0.0, **kw):
    J = len(JOINTS)
    ndof = J + 2
    cfg = vs.VicSteeringConfig(vic_joints=JOINTS, vic_law="v4", delta_tau_up_s=0.05, delta_tau_down_s=0.15,
                               frontal_human_ref=True, **kw)
    sim = SimpleNamespace(_common_p_gains=torch.zeros(n, ndof), _common_d_gains=torch.zeros(n, ndof),
                          vic_ff_prior=torch.zeros(n, ndof), vic_ff_res_target=torch.zeros(n, ndof),
                          vic_ff_res=torch.zeros(n, ndof), data_conversion=None,
                          human_active_torques=torch.zeros(n, ndof))
    sim._human_joint_model = SimpleNamespace(
        a=torch.zeros(1, ndof), offset=torch.full((1,), -50.0), energy_scale=torch.zeros(1), exp_cap=1e6,
        rest=torch.zeros(ndof), k_negative=torch.full((ndof,), k_pass), k_positive=torch.full((ndof,), k_pass),
        spring_cap=1e9, damping=torch.zeros(ndof))
    sim.q, sim.qd = torch.full((n, ndof), 0.1), torch.zeros(n, ndof)
    sim._get_simulator_dof_state = lambda: _Dof(sim.q, sim.qd)
    env = SimpleNamespace(simulator=sim, dt=1 / 30, progress_buf=torch.full((n,), 10))
    self = object.__new__(vs.VicSteering)
    self.config, self.env = cfg, env
    self._idx = torch.arange(J)
    self._leg = torch.tensor([0 if j.startswith("L_") else 1 for j in JOINTS])
    self._k_old = torch.full((J,), 100.0)
    self._b_old = torch.full((J,), 10.0)
    self._tau_max = torch.full((J,), 150.0)
    self._tar_speed = torch.full((n,), 1.0)
    self.alpha, self.beta = 1.0, 0.0
    self._gain_log_path = None
    self._write_gains = lambda: None
    self._log = lambda: None
    self._sag = torch.arange(6)
    self._bw = 63.3 * 9.81
    self._mean_k = torch.zeros(3, J)
    ranges = dict(vs.HUMAN_RANGES, **vs.HUMAN_RANGES_FRONTAL)
    self._lk_h, self._half = torch.zeros(J, 3), torch.zeros(J, 3)
    for i, j in enumerate(JOINTS):
        for s in range(3):
            lo, hi = ranges[j[2:]]["k"][s]
            self._lk_h[i, s] = 0.5 * (math.log(lo) + math.log(hi))
            self._half[i, s] = 0.5 * (math.log(hi) - math.log(lo)) + math.log(cfg.band_widen)
    self._v4_leg_load = lambda contact: sim.load
    self._init_v4(n, torch.device("cpu"), ranges)
    return self, sim


def _step4(self, sim, a, fresh=False, root_v=1.0):
    n, J = a.shape
    self._pre_physics_v4(torch.zeros(n, J + 2), torch.tanh(a), torch.zeros(n, J), None,
                         torch.full((n,), root_v), torch.full((n,), fresh))
    return sim._common_p_gains[:, :J].clone(), sim._common_d_gains[:, :J].clone()


def test_v4_centre_at_zero_action_and_passive_subtracted():
    J = len(JOINTS)
    for kp_pass in (0.0, 10.0):
        self, sim = _stub(k_pass=kp_pass, v4_late_anchor=False)
        sim.load = torch.tensor([[1.0, 1.0]]).expand(4, 2)                   # walking, both loaded: stance
        kp, kd = _step4(self, sim, torch.zeros(4, J), fresh=True)
        centre = self._v4_lk[:, 1].exp()
        want = torch.maximum(centre - kp_pass, 0.02 * centre)
        assert torch.allclose(kp[0], want, rtol=1e-4)
        assert self._band[0] == 0 and self._imp[0] == pytest.approx(1.0)    # total inside the human range
        assert torch.allclose(kd[0], self._v4_lb[:, 1].exp(), rtol=1e-4)     # b at its centre (b_pass = 0)


def test_v4_reward_only_outside_range_and_standing_blend():
    J = len(JOINTS)
    self, sim = _stub(v4_late_anchor=False)
    sim.load = torch.tensor([[1.0, 1.0]]).expand(4, 2)
    for _ in range(60):
        _step4(self, sim, torch.full((4, J), 5.0))                          # a -> +1: twice the range top
    assert self._band[0] > 0 and self._imp[0] < 1.0
    hi = torch.tensor([vs.HUMAN_RANGES_FRONTAL.get(j[2:], vs.HUMAN_RANGES.get(j[2:]))["k"][1][1] for j in JOINTS])
    k_eff = sim._common_p_gains[0, :J]
    assert torch.allclose(k_eff, 2.0 * hi, rtol=1e-3)
    # Slow command and body with both feet loaded: the reference moves to the standing range.
    self, sim = _stub(v4_late_anchor=False)
    self._tar_speed[:] = 0.0
    sim.load = torch.tensor([[0.5, 0.5]]).expand(4, 2)
    kp, _ = _step4(self, sim, torch.zeros(4, J), fresh=True, root_v=0.0)
    ank = JOINTS.index("L_Ankle_y")
    assert kp[0, ank] == pytest.approx(math.sqrt(290 * 340), rel=1e-4)


def test_v4_ankle_late_stance_anchor():
    J = len(JOINTS)
    self, sim = _stub()
    sim.load = torch.tensor([[1.0, 0.0]]).expand(4, 2)                        # left leg in single stance
    ank = JOINTS.index("L_Ankle_y")
    kp0, _ = _step4(self, sim, torch.zeros(4, J), fresh=True)
    assert kp0[0, ank] == pytest.approx(math.sqrt(22 * 406), rel=1e-4)
    for _ in range(int(0.6 / (1 / 30))):                                    # 0.6 s of stance vs T0 0.65 s
        kp, _ = _step4(self, sim, torch.zeros(4, J))
    assert kp[0, ank] == pytest.approx(math.sqrt(133 * 234), rel=1e-3)
    assert kp[0, JOINTS.index("L_Knee_y")] == pytest.approx(kp0[0, JOINTS.index("L_Knee_y")])


def test_v4_coactivation_logged_for_j2m():
    J = len(JOINTS)
    self, sim = _stub(v4_late_anchor=False)
    self._tar_speed[:] = 0.0
    sim.load = torch.tensor([[0.5, 0.5]]).expand(4, 2)
    sim.human_active_torques[:] = 8.0
    kp, _ = _step4(self, sim, torch.zeros(4, J), fresh=True, root_v=0.0)
    ank = JOINTS.index("L_Ankle_y")
    want = (kp[0, ank] - 3.5 * 8.0) / 3.5 / 150.0
    assert self.coactivation_level()[0, ank] == pytest.approx(float(want), rel=1e-4)
