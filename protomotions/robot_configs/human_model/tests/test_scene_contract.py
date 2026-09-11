# SPDX-License-Identifier: Apache-2.0
"""Execute the native SceneCfg constructor using CPU configuration stand-ins.

The complete constructor comes from the patched source. Only IsaacLab's config
containers are replaced; this verifies its force and asset settings, not PhysX.
"""

import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from protomotions.robot_configs.base import ControlType
from protomotions.robot_configs.human_model.integration import prepare_simulator
from protomotions.robot_configs.smpl import SmplRobotConfig


class Config:
    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)

    def replace(self, **kwargs):
        values = dict(vars(self))
        values.update(kwargs)
        return type(self)(**values)


Config.InitialStateCfg = Config


class Implicit(Config):
    pass


class Explicit(Config):
    pass


def native_scene_class():
    source = Path(__file__).resolve().parents[3] / "simulator/isaaclab/utils/scene.py"
    tree = ast.parse(source.read_text(), filename=str(source))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SceneCfg")
    cls.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls], type_ignores=[])
    ast.fix_missing_locations(module)
    scope = {
        "InteractiveSceneCfg": Config,
        "AssetBaseCfg": Config,
        "ArticulationCfg": Config,
        "ContactSensorCfg": Config,
        "ImplicitActuatorCfg": Implicit,
        "IdealPDActuatorCfg": Explicit,
        "ISAAC_NUCLEUS_DIR": "unused-cpu-test",
        "ControlType": ControlType,
        "sim_utils": SimpleNamespace(**{name: Config for name in (
            "DomeLightCfg", "UsdFileCfg", "RigidBodyPropertiesCfg",
            "ArticulationRootPropertiesCfg", "CollisionPropertiesCfg", "PreviewSurfaceCfg",
        )}),
    }
    exec(compile(module, str(source), "exec"), scope)
    return scope["SceneCfg"]


@pytest.mark.parametrize("profile,command", [
    ("healthy_adult_v1", ControlType.BUILT_IN_PD),
    ("legacy", ControlType.BUILT_IN_PD),
    ("legacy", ControlType.TORQUE),
])
def test_native_scene_preserves_active_commands_and_selects_backend_forces(profile, command, monkeypatch):
    monkeypatch.setenv("PROTOMOTIONS_HUMAN_MODEL", profile)
    robot = SmplRobotConfig()
    robot.control.control_type = command
    before = deepcopy(robot.control)
    cfg = SimpleNamespace(
        _target_="protomotions.simulator.isaaclab.simulator.IsaacLabSimulator",
        sim=SimpleNamespace(decimation=4, physx=SimpleNamespace(
            max_depenetration_velocity=10.0, num_position_iterations=4,
            num_velocity_iterations=1, contact_offset=0.02, rest_offset=0.0,
        )),
    )
    backend, model = prepare_simulator(robot, cfg, torch.device("cpu"))
    scene = native_scene_class()(cfg, backend)
    assert robot.control.control_type == before.control_type == backend.control.control_type
    for name, actuator in scene.robot.actuators.items():
        assert robot.control.control_info[name].stiffness == before.control_info[name].stiffness
        assert robot.control.control_info[name].damping == before.control_info[name].damping
        if model is not None:
            assert isinstance(actuator, Explicit)
            assert actuator.stiffness == actuator.damping == 0
            assert actuator.effort_limit == actuator.effort_limit_sim
            assert actuator.effort_limit == backend.control.control_info[name].effort_limit
        elif command == ControlType.BUILT_IN_PD:
            assert isinstance(actuator, Implicit)
            assert actuator.stiffness == before.control_info[name].stiffness
            assert actuator.damping == before.control_info[name].damping
            assert not hasattr(actuator, "effort_limit")
        else:
            assert isinstance(actuator, Explicit)
            assert actuator.stiffness == actuator.damping == 0
    assert scene.robot.spawn.usd_path == f"{backend.asset.asset_root}/{backend.asset.usd_asset_file_name}"
    if model is not None:
        assert Path(scene.robot.spawn.usd_path).is_file()
        assert scene.robot.spawn.usd_path.endswith("smpl_humanoid_healthy_adult_v1.usda")
