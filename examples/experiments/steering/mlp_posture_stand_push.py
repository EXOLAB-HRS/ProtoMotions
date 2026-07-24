"""⑥ stand-only environment: static reference, zero command, modest pushes."""
import importlib.util
from pathlib import Path

from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.control.steering_control import SteeringControlConfig
from protomotions.simulator.base_simulator.config import (
    DomainRandomizationConfig, PushDomainRandomizationConfig,
)
from human_controller.posture import compute_posture_observation

_BASE_PATH = Path(__file__).with_name("mlp_locomotion_omni_uniform_speed_s80_task70_foot05.py")
_SPEC = importlib.util.spec_from_file_location("posture_stand_base", _BASE_PATH)
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

terrain_config = _BASE.terrain_config
scene_lib_config = _BASE.scene_lib_config
motion_lib_config = _BASE.motion_lib_config
agent_config = _BASE.agent_config
apply_inference_overrides = _BASE.apply_inference_overrides


def env_config(robot_cfg, args):
    cfg = _BASE.env_config(robot_cfg, args)
    cfg.control_components["steering"] = SteeringControlConfig(
        tar_speed_min=0.0, tar_speed_max=0.0, stop_probability=1.0,
        enable_rand_facing=False, heading_change_steps_min=100000,
        heading_change_steps_max=100001,
    )
    cfg.observation_components["posture_obs"] = MdpComponent(
        compute_func=compute_posture_observation,
        dynamic_vars={
            "root_rot": EnvContext.current.root_rot,
            "root_ang_vel": EnvContext.current.root_ang_vel,
            "root_pos": EnvContext.current.root_pos,
            "ground_height": EnvContext.ground_heights,
        },
    )
    cfg.max_episode_length = 240
    return cfg


def configure_robot_and_simulator(robot_cfg, simulator_cfg, args):
    """Keep the perturbation mechanism in simulator configuration, not EnvConfig."""
    simulator_cfg.domain_randomization = DomainRandomizationConfig(
        push=PushDomainRandomizationConfig(
            push_interval_range=(1.25, 2.5),
            max_linear_velocity=(0.30, 0.30, 0.0),
            max_angular_velocity=(0.25, 0.25, 0.15),
        )
    )
