# SPDX-License-Identifier: Apache-2.0
"""Physics and integration checks, runnable on CPU without IsaacSim."""

import copy
import os
from pathlib import Path
import re
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from protomotions.robot_configs.human_model.assets import render_mjcf, render_usda
from protomotions.robot_configs.human_model.dynamics import HumanJointModel
from protomotions.robot_configs.human_model.integration import prepare_simulator
from protomotions.robot_configs.human_model.profile import load_profile
from protomotions.robot_configs.smpl import SmplRobotConfig
from protomotions.robot_configs.base import ControlType
from protomotions.simulator.mujoco.config import MujocoSimulatorConfig

ASSETS = Path(__file__).resolve().parents[3] / "data/assets"


@pytest.fixture
def model():
    p = load_profile()
    return HumanJointModel(p, list(p["joints"]), dtype=torch.float64)


@pytest.fixture
def robot():
    return SmplRobotConfig()


def config():
    return MujocoSimulatorConfig(headless=True, num_envs=1, experiment_name="human_model_test")


@pytest.mark.parametrize("name", ["../smpl", "unknown", ""])
def test_profile_rejects_unknown_or_path_names(name):
    with pytest.raises(ValueError):
        load_profile(name)


def test_reference_values_and_signed_rom(model):
    i = model.names.index("L_Knee_y")
    assert model.lower[i].item() == pytest.approx(np.deg2rad(-1.3))
    assert model.upper[i].item() == pytest.approx(np.deg2rad(139.8))
    assert model.negative[i] == 149.0  # knee extension strength in Nm
    assert model.positive[i] == 93.1
    assert model.damping[model.names.index("L_Wrist_x")] == pytest.approx((1.89 + 2.56) / 2)


@pytest.mark.parametrize("scale", [0.1, 1.0, 10.0])
def test_elastic_force_is_negative_energy_gradient_even_at_extrapolated_poses(model, scale):
    generator = torch.Generator().manual_seed(43)
    q = (torch.randn(7, 69, generator=generator, dtype=torch.float64) * scale).requires_grad_()
    grad, = torch.autograd.grad(model.potential_energy(q).sum(), q)
    torch.testing.assert_close(model.elastic_torque(q), -grad, atol=1e-9, rtol=1e-9)
    assert torch.isfinite(grad).all()


def test_damping_never_injects_energy(model):
    qd = torch.linspace(-1000, 1000, 69, dtype=torch.float64).repeat(3, 1)
    assert (model.damping_torque(qd) * qd <= 0).all()
    assert torch.count_nonzero(model.damping_torque(qd)) > 0


def test_directional_active_caps_do_not_clip_passive_torques(model):
    q = (model.lower + model.upper).expand(4, -1) / 2
    qd = torch.zeros_like(q)
    command = torch.full_like(q, 1e6)
    total, active, passive = model.torques(command, q, qd)
    torch.testing.assert_close(active, model.positive.expand_as(q))
    torch.testing.assert_close(total, active + passive)
    assert (passive.abs() > 0.1).any()
    torch.testing.assert_close(model.limit_active(-command), -model.negative.expand_as(q))
    assert (total.abs() <= model.backend_limit + 1e-9).all()


def test_passive_coupling_and_plantarflexor_sign(model):
    q = torch.zeros(69, dtype=torch.float64)
    hip, knee, ankle = [model.names.index(f"L_{n}_y") for n in ("Hip", "Knee", "Ankle")]
    before = model.elastic_torque(q)
    q[knee] = 0.5
    after = model.elastic_torque(q)
    assert after[hip] != before[hip]  # RF/hamstrings biarticular coupling
    q[ankle] = -0.2  # dorsiflexion stretches passive plantarflexors
    assert model.elastic_torque(q)[ankle] > after[ankle] > 0


def test_backend_force_ceiling_covers_extreme_extrapolation(model):
    q = torch.randn(100, 69, dtype=torch.float64) * 100
    qd = torch.randn_like(q) * 1000
    total, _, _ = model.torques(torch.randn_like(q) * 1e6, q, qd)
    assert torch.isfinite(total).all()
    assert (total.abs() <= model.backend_limit + 1e-9).all()


def test_asset_preserves_mass_geometry_and_dof_order(model):
    source = (ASSETS / "mjcf/smpl_humanoid.xml").read_text()
    limits = dict(zip(model.names, model.backend_limit.tolist()))
    derived = render_mjcf(source, load_profile(), limits)
    old, new = mujoco.MjModel.from_xml_string(source), mujoco.MjModel.from_xml_string(derived)
    assert new.nv == old.nv == 75 and new.nu == old.nu == 69
    for attr in ("body_mass", "body_inertia", "body_pos", "geom_pos", "geom_size", "jnt_axis", "jnt_type"):
        np.testing.assert_array_equal(getattr(old, attr), getattr(new, attr))
    assert old.names == new.names
    for i, name in enumerate(model.names):
        joint = mujoco.mj_name2id(new, mujoco.mjtObj.mjOBJ_JOINT, name)
        np.testing.assert_allclose(new.jnt_range[joint], [model.lower[i], model.upper[i]])
        assert new.jnt_stiffness[joint] == 0
    assert (new.dof_damping == 0).all()
    assert (new.actuator_gear[:, 0] == 1).all()


@pytest.mark.parametrize("side,sign", [("L", 1), ("R", -1)])
def test_anatomical_signs_by_forward_kinematics(side, sign):
    m = mujoco.MjModel.from_xml_path(str(ASSETS / "mjcf/smpl_humanoid.xml"))
    d = mujoco.MjData(m)

    def setq(name, q):
        d.qpos[m.jnt_qposadr[m.joint(name).id]] = q

    # Hip flexion moves the ankle forward; knee flexion moves it backward.
    mujoco.mj_forward(m, d)
    x0 = d.xpos[m.body(f"{side}_Ankle").id, 0]
    setq(f"{side}_Hip_y", -0.3)
    mujoco.mj_forward(m, d)
    assert d.xpos[m.body(f"{side}_Ankle").id, 0] > x0
    setq(f"{side}_Hip_y", 0)
    setq(f"{side}_Knee_y", 0.3)
    mujoco.mj_forward(m, d)
    assert d.xpos[m.body(f"{side}_Ankle").id, 0] < x0
    # T-pose -> arm down; mirrored elbow flexion moves the wrist forward.
    setq(f"{side}_Shoulder_x", -sign * np.pi / 2)
    mujoco.mj_forward(m, d)
    x0 = d.xpos[m.body(f"{side}_Wrist").id, 0]
    setq(f"{side}_Elbow_z", -sign * 0.3)
    mujoco.mj_forward(m, d)
    assert d.xpos[m.body(f"{side}_Wrist").id, 0] > x0


def test_frozen_checkpoint_reapplies_metadata_without_changing_pd_or_action_semantics(robot, tmp_path, monkeypatch):
    monkeypatch.setenv("PROTOMOTIONS_HUMAN_MODEL_CACHE", str(tmp_path))
    # Unpickling old RobotConfig objects does not call __post_init__.
    robot.__dict__.pop("human_model_profile", None)
    robot.kinematic_info.dof_limits_lower.fill_(-np.pi)
    robot.kinematic_info.dof_limits_upper.fill_(np.pi)
    for row in robot.control.control_info.values():
        row.effort_limit = 500
    before = copy.deepcopy(robot)
    backend, model = prepare_simulator(robot, config(), torch.device("cpu"))
    assert model is not None
    assert robot.control.control_type == before.control.control_type == backend.control.control_type
    assert robot.asset == before.asset
    assert robot.kinematic_info.dof_names == before.kinematic_info.dof_names
    assert not torch.equal(robot.kinematic_info.dof_limits_lower, before.kinematic_info.dof_limits_lower)
    for name in model.names:
        assert backend.control.control_info[name].stiffness == before.control.control_info[name].stiffness
        assert backend.control.control_info[name].damping == before.control.control_info[name].damping
    assert Path(backend.asset.asset_root, backend.asset.asset_file_name).is_file()


def test_legacy_and_other_robots_are_unmodified(robot, monkeypatch):
    monkeypatch.setenv("PROTOMOTIONS_HUMAN_MODEL", "legacy")
    result, force = prepare_simulator(robot, config(), torch.device("cpu"))
    assert result is robot and force is None
    monkeypatch.delenv("PROTOMOTIONS_HUMAN_MODEL")
    other = SimpleNamespace(asset=SimpleNamespace(asset_file_name="mjcf/g1.xml"))
    result, force = prepare_simulator(other, config(), torch.device("cpu"))
    assert result is other and force is None


def test_usd_lfs_pointer_fails_instead_of_silently_ignoring_model(model):
    with pytest.raises(ValueError, match="Git LFS"):
        render_usda("version https://git-lfs.github.com/spec/v1\n", load_profile(), {})


def test_usda_parser_updates_all_axes_and_preserves_other_properties(model):
    # Minimal schema fixture; actual upstream USDA is verified separately when
    # LFS assets are available (test accepts HUMAN_MODEL_TEST_USDA below).
    profile = load_profile()
    parts = ["#usda 1.0\n"]
    for body in dict.fromkeys(n.rsplit("_", 1)[0] for n in model.names):
        parts.append(f'        def PhysicsJoint "{body}" (\n        )\n        {{\n')
        parts.append('            rel physics:body0 = </preserve/me>\n')
        for a in "xyz":
            token = f"rot{a.upper()}"
            parts.append(f'            custom token mjcf:{token}:name = "{body}_{a}"\n')
            for prop in (f"limit:{token}:physics:low", f"limit:{token}:physics:high", f"drive:{token}:physics:stiffness", f"drive:{token}:physics:damping", f"drive:{token}:physics:maxForce", f"physxLimit:{token}:stiffness", f"physxLimit:{token}:damping"):
                parts.append(f"            float {prop} = 500\n")
        parts.append("        }\n")
    result = render_usda("".join(parts), profile, dict(zip(model.names, model.backend_limit.tolist())))
    assert result.count("</preserve/me>") == 23
    assert len(re.findall(r"physics:low =", result)) == 69
    assert "physics:low = -132.1" in result
    with pytest.raises(ValueError, match="mapping changed"):
        render_usda("".join(parts).replace('name = "L_Hip_x"', 'name = "bad"'), profile, dict(zip(model.names, model.backend_limit.tolist())))


def test_actual_usda_with_openusd(model):
    path = Path(os.environ.get("HUMAN_MODEL_TEST_USDA", str(ASSETS / "usd/smpl_humanoid.usda")))
    source = path.read_text()
    if source.startswith("version https://git-lfs.github.com/spec"):
        pytest.skip("Actual USDA requires Git LFS; optionally set HUMAN_MODEL_TEST_USDA")
    Usd = pytest.importorskip("pxr.Usd")
    Sdf = pytest.importorskip("pxr.Sdf")
    output = render_usda(source, load_profile(), dict(zip(model.names, model.backend_limit.tolist())))
    layers = [Sdf.Layer.CreateAnonymous("test.usda") for _ in range(2)]
    for layer, text in zip(layers, [source, output]):
        assert layer.ImportFromString(text)
    old, new = [Usd.Stage.Open(layer) for layer in layers]
    assert [p.GetPath() for p in old.Traverse()] == [p.GetPath() for p in new.Traverse()]
    count = 0
    for prim in old.Traverse():
        updated = new.GetPrimAtPath(prim.GetPath())
        for attr in prim.GetAttributes():
            name = attr.GetName()
            if prim.GetTypeName() == "PhysicsJoint" and name.startswith(("drive:rot", "limit:rot", "physxLimit:rot")):
                continue
            assert attr.Get() == updated.GetAttribute(name).Get()
        if prim.GetTypeName() == "PhysicsJoint":
            for axis in "xyz":
                i = model.names.index(f"{prim.GetName()}_{axis}")
                token = f"rot{axis.upper()}"
                assert updated.GetAttribute(f"limit:{token}:physics:low").Get() == pytest.approx(float(model.lower[i]) * 180 / np.pi)
                assert updated.GetAttribute(f"limit:{token}:physics:high").Get() == pytest.approx(float(model.upper[i]) * 180 / np.pi)
                assert updated.GetAttribute(f"drive:{token}:physics:stiffness").Get() == 0
                assert updated.GetAttribute(f"drive:{token}:physics:damping").Get() == 0
                count += 1
    assert count == 69


def test_mujoco_runtime_uses_passive_forces_at_every_substep(robot):
    from protomotions.components.scene_lib import SceneLib, SceneLibConfig
    from protomotions.simulator.mujoco.simulator import MujocoSimulator

    sim = MujocoSimulator(config(), robot, None, torch.device("cpu"), SceneLib(SceneLibConfig()))
    sim._initialize_with_markers(None)
    assert sim.control_type == ControlType.TORQUE
    assert sim.robot_config.control.control_type == ControlType.BUILT_IN_PD
    sim.model.opt.gravity[:] = 0
    # Place the floating model clear of ground, disable contact for this force test.
    sim.data.qpos[2] = 3
    sim.model.geom_contype[:] = 0
    sim.model.geom_conaffinity[:] = 0
    sim._common_actions.zero_()
    mujoco.mj_forward(sim.model, sim.data)
    for _ in range(10):
        sim._physics_step()
    assert np.isfinite(sim.data.qpos).all() and np.isfinite(sim.data.qvel).all()
    assert sim.data.time == pytest.approx(10 * sim.decimation / sim.config.sim.fps)
    assert torch.count_nonzero(sim.human_passive_torques) > 0
    hm = sim._human_joint_model
    assert (sim.human_active_torques <= hm.positive + 1e-5).all()
    assert (sim.human_active_torques >= -hm.negative - 1e-5).all()
    assert (np.abs(sim.data.qvel) < 100).all()
