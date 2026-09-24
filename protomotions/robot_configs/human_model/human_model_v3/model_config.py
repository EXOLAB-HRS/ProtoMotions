"""Version 3 strength candidate; reuses v2 geometry and independently fixed PD."""
from pathlib import Path
import json

MODEL = {
    'model_id': 'human_model_v3', 'profile_id': 'human_model_v3',
    'status': 'active', 'runnable': True,
    'assets': {'usd': '../human_model_v2/assets/human_model_v2.usda',
               'mjcf': '../human_model_v2/assets/human_model_v2.xml'},
    'profile': 'profiles/healthy_adult_v3.json',
    'kinematic_model_id': 'human_model_v2',
}


def robot_config():
    from ..human_model_v2.model_config import robot_config as base_robot
    from ..common.integration import apply_metadata
    robot = base_robot()
    robot.human_model_profile = 'human_model_v3'
    configure_strength_profile(robot)
    _configure_gains(robot)
    apply_metadata(robot)
    return robot


def configure_strength_profile(robot, strength_profile='reference'):
    """Low/high are explicit sensitivity scenarios, NOT population percentiles."""
    from ..common.integration import apply_metadata
    if robot.human_model_profile != 'human_model_v3':
        raise ValueError('Requires human_model_v3')
    if strength_profile not in ('low','reference','high'):
        raise ValueError('Unknown strength scenario')
    settings=json.loads((Path(__file__).parent/'profiles/strength_scenarios.json').read_text())
    robot.human_model_parameters=settings[strength_profile]['parameters']
    apply_metadata(robot)


def _configure_gains(robot):
    from protomotions.robot_configs.base import ControlType
    settings=json.loads((Path(__file__).parent/'profiles/strength_scenarios.json').read_text())
    robot.control.control_type=ControlType.PROPORTIONAL
    for name, gains in settings['pd_gains'].items():
        robot.control.control_info[name].stiffness=gains['kp']
        # Reset from canonical values: repeated experiment configuration must
        # not multiply the selected revision's damping a second time.
        multiplier = (2.0 if getattr(robot, 'human_model_collision_profile', None) == 'human_model_v3.1'
                      and name.startswith(('Neck_', 'Head_')) else 1.0)
        robot.control.control_info[name].damping=gains['kd'] * multiplier


def configure_pd(robot, simulator):
    """PD selected on v2 trunk assay; identical gains for all strength scenarios."""
    _configure_gains(robot)
    simulator.sim.fps=480
    simulator.sim.decimation=16
