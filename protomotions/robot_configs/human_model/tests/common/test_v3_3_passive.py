"""human_model_v3.3 (2026-11-02): v3.1 plus the Silder-curve-calibrated passive leg exponentials."""
import math

import torch

from protomotions.robot_configs.factory import robot_config
from protomotions.robot_configs.human_model.common.dynamics import HumanJointModel
from protomotions.robot_configs.human_model.common.integration import selected_model_parameters
from protomotions.robot_configs.human_model.common.profile import load_profile


def _model(name):
    r = robot_config(name)
    return r, HumanJointModel(load_profile(r.human_model_profile), r.kinematic_info.dof_names, **selected_model_parameters(r))


def test_v3_3_is_v3_1_with_new_passive_legs():
    r31, m31 = _model("human_model_v3.1")
    r33, m33 = _model("human_model_v3.3")
    assert r33.human_model_collision_profile == r31.human_model_collision_profile and r33.asset.self_collisions
    assert {n: (c.stiffness, c.damping) for n, c in r33.control.control_info.items()} == \
           {n: (c.stiffness, c.damping) for n, c in r31.control.control_info.items()}
    assert torch.equal(m33.positive, m31.positive) and torch.equal(m33.negative, m31.negative)
    assert "L_gastrocnemius" in m33.passive_component_names and "L_gastrocnemius" not in m31.passive_component_names
    q = torch.zeros(1, len(m33.names)); k = m33.names.index("L_Knee_y"); q[0, k] = math.radians(60)
    assert abs(float(m31.elastic_torque(q)[0, k]) + 16.58) < 0.1          # v3: -16.6 N m at hip 0 / knee 60 deg
    assert -3.0 < float(m33.elastic_torque(q)[0, k]) < -1.0               # v3.3: a few N m (Silder curves)


def test_v3_3_passive_is_conservative():
    _, m = _model("human_model_v3.3")
    g = torch.Generator().manual_seed(0)
    q = (m.lower + (m.upper - m.lower) * torch.rand(256, len(m.names), generator=g)).double().requires_grad_(True)
    m64 = HumanJointModel(load_profile("human_model_v3"), m.names, dtype=torch.float64,
                          **selected_model_parameters(robot_config("human_model_v3.3")))
    e = m64.potential_energy(q).sum(); (grad,) = torch.autograd.grad(e, q)
    assert torch.allclose(m64.elastic_torque(q.detach()), -grad, atol=1e-6)
