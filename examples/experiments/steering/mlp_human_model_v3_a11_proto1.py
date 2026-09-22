"""a11-proto1: fresh full-v4 omnidirectional AMP teacher candidate."""

import importlib.util
from pathlib import Path


_SPEC = importlib.util.spec_from_file_location(
    "a11_omni_base", Path(__file__).with_name("mlp_human_model_v2_omni_turn_stop.py")
)
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

EXPERIMENT_ID = "a11-proto1"
STOP_PROBABILITY = 0.0
HEAD_REWARD_WEIGHT = 0.0

terrain_config = _BASE.terrain_config
scene_lib_config = _BASE.scene_lib_config
motion_lib_config = _BASE.motion_lib_config
apply_inference_overrides = _BASE.apply_inference_overrides


def configure_robot_and_simulator(robot_cfg, simulator_cfg, args):
    from protomotions.robot_configs.human_model.human_model_v3.model_config import configure_pd
    if robot_cfg.human_model_profile != "human_model_v3":
        raise ValueError("a11-proto1 requires --robot-name human_model_v3")
    configure_pd(robot_cfg, simulator_cfg)
    robot_cfg.contact_bodies = (
        robot_cfg.common_naming_to_robot_body_names["all_left_foot_bodies"]
        + robot_cfg.common_naming_to_robot_body_names["all_right_foot_bodies"]
    )


def env_config(robot_cfg, args):
    cfg = _BASE.env_config(robot_cfg, args)
    steering = cfg.control_components["steering"]
    steering.tar_speed_min = 0.5
    steering.tar_speed_max = 1.5
    steering.stop_probability = STOP_PROBABILITY
    steering.random_heading_probability = 1.0
    steering.random_speed_probability = 1.0
    steering.heading_change_steps_min = 150
    steering.heading_change_steps_max = 301
    steering.enable_rand_facing = True
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    cfg = _BASE.agent_config(robot_cfg, env_cfg, args)
    cfg.task_reward_w = 0.6
    cfg.amp_parameters.discriminator_reward_w = 0.4
    cfg.model.discriminator_optimizer.lr = 1e-4
    cfg.amp_parameters.discriminator_batch_size = min(4096, args.batch_size)
    return cfg
