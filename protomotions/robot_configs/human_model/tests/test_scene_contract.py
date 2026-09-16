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
from protomotions.robot_configs.human_model.integration import prepare_simulator, selected_model_parameters
from protomotions.robot_configs.smpl import SmplRobotConfig


def test_com_randomization_uses_physical_names_with_intermediate_frame_links():
    source=Path(__file__).resolve().parents[3]/'simulator/isaaclab/simulator.py'
    tree=ast.parse(source.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='IsaacLabSimulator')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_apply_domain_randomization_if_needed')
    scope={'torch':torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(source),'exec'),scope)
    original=torch.zeros((1,4,7));written=[]
    view=SimpleNamespace(get_coms=lambda:original,set_coms=lambda values,ids:written.append(values.clone()))
    native_names=['Pelvis','_joint_frame_Knee_x','_joint_frame_Knee_y','Knee']
    robot=SimpleNamespace(root_physx_view=view,find_bodies=lambda names,preserve_order:([native_names.index(n) for n in names],names))
    obj=SimpleNamespace(config=SimpleNamespace(num_envs=1),_robot=robot,
        robot_config=SimpleNamespace(kinematic_info=SimpleNamespace(body_names=['Pelvis','Knee'])),
        _domain_randomization={'center_of_mass':{'body_indices':[1],'com':torch.tensor([[[.1,.2,.3]]])}},
        _apply_scene_object_properties_after_spawn=lambda ids:None)
    scope['_apply_domain_randomization_if_needed'](obj)
    torch.testing.assert_close(written[0][0,3,:3],torch.tensor([.1,.2,.3]))
    assert not written[0][0,:3].any() and not original.any()


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


def test_fatigue_mapping_reaches_backend_and_allocates_replay_state(monkeypatch):
    monkeypatch.delenv('PROTOMOTIONS_HUMAN_MODEL_FEATURES',raising=False)
    robot=SmplRobotConfig(human_model_parameters={'features':['fatigue'],'fatigue_regions':{'R_Knee_y':'Knee'}})
    cfg=SimpleNamespace(_target_='protomotions.simulator.isaaclab.simulator.IsaacLabSimulator',num_envs=2)
    _,model=prepare_simulator(robot,cfg,torch.device('cpu'))
    assert model._fatigue.shape==(2,69,2,3)
    assert (model._fatigue[...,0]==1).all() and not model._fatigue[...,1:].any()
    assert model.fatigue_regions=={'R_Knee_y':'Knee'}


def test_serialized_plant_parameters_reach_native_actuator_limits(monkeypatch):
    monkeypatch.delenv('PROTOMOTIONS_HUMAN_MODEL_FEATURES',raising=False)
    parameters={'features':['strength','activation'], 'strength_cohort':'male',
                'strength_reference_size':[79,1.8], 'active_strength_scale':{'L_Ankle_y':[1,1.75]}}
    robot=SmplRobotConfig(human_model_parameters=parameters)
    cfg=SimpleNamespace(_target_='protomotions.simulator.isaaclab.simulator.IsaacLabSimulator',num_envs=2)
    before=deepcopy(robot.human_model_parameters)
    backend,model=prepare_simulator(robot,cfg,torch.device('cpu'))
    i=model.names.index('L_Ankle_y')
    assert model.strength_cohort=='male' and model.strength_reference_size==(79.,1.8)
    assert model.active_strength_scale[i,1]==1.75
    assert model._activation.shape==(2,69,2)
    assert backend.control.control_info['L_Ankle_y'].effort_limit==float(model.backend_limit[i])
    assert robot.control.control_info['L_Ankle_y'].effort_limit==max(float(model.negative[i]),float(model.positive[i]))
    assert robot.human_model_parameters==before
    backend.human_model_parameters['active_strength_scale']['L_Ankle_y'][1]=2
    assert robot.human_model_parameters==before
    monkeypatch.setenv('PROTOMOTIONS_HUMAN_MODEL_FEATURES','strength')
    assert selected_model_parameters(robot)['features']==('strength',)
    with pytest.raises(ValueError,match='unsupported'):
        SmplRobotConfig(human_model_parameters={'typo':1})
    with pytest.raises(ValueError,match='sequence'):
        monkeypatch.delenv('PROTOMOTIONS_HUMAN_MODEL_FEATURES')
        SmplRobotConfig(human_model_parameters={'features':'strength'})


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
@pytest.mark.parametrize("fix_base", [None, True, False])
def test_native_scene_preserves_active_commands_and_selects_backend_forces(profile, command, monkeypatch, fix_base):
    monkeypatch.setenv("PROTOMOTIONS_HUMAN_MODEL", profile)
    robot = SmplRobotConfig()
    robot.control.control_type = command
    robot.asset.fix_base_link = fix_base
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
    assert scene.robot.spawn.articulation_props.fix_root_link is fix_base
    assert scene.robot.spawn.usd_path == f"{backend.asset.asset_root}/{backend.asset.usd_asset_file_name}"
    if model is not None:
        assert Path(scene.robot.spawn.usd_path).is_file()
        assert backend.human_model_usd_joint_mode == 'serial'
        assert scene.robot.spawn.usd_path.endswith('smpl_humanoid_serial.usda')
        assert Path(scene.robot.spawn.usd_path).read_text().count('def PhysicsRevoluteJoint') == 69
