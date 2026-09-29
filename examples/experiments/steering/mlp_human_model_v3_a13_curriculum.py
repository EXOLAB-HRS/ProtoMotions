"""Shared A7 warm-start curriculum for a13 stages B, C and D."""

import importlib.util
from pathlib import Path

from protomotions.envs.control.continuous_steering import ContinuousSteeringConfig


_SPEC = importlib.util.spec_from_file_location(
    "a13_base", Path(__file__).with_name("mlp_human_model_v3_a12_scratch_head.py")
)
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

EXPERIMENT_ID = "a13"
HEAD_REWARD_WEIGHT = _BASE.HEAD_REWARD_WEIGHT

STAGES = {
    "b": dict(
        fixed_fraction=0.30, turn_fraction=0.0, stop_probability=0.0,
        tar_speed_min=0.7, tar_speed_max=1.4,
        acceleration_min=0.4, acceleration_max=0.8,
        full_heading_for_turn=False, independent_facing_fraction=0.0,
    ),
    "c": dict(
        fixed_fraction=0.20, turn_fraction=0.0, stop_probability=0.50,
        tar_speed_min=0.7, tar_speed_max=1.4,
        acceleration_min=0.4, acceleration_max=0.8,
        full_heading_for_turn=False, independent_facing_fraction=0.0,
    ),
    "d": dict(
        fixed_fraction=0.20, turn_fraction=0.60, stop_probability=0.0,
        tar_speed_min=0.7, tar_speed_max=1.4,
        acceleration_min=0.4, acceleration_max=0.8,
        full_heading_for_turn=True, independent_facing_fraction=1.0,
    ),
}

terrain_config = _BASE.terrain_config
scene_lib_config = _BASE.scene_lib_config
motion_lib_config = _BASE.motion_lib_config
agent_config = _BASE.agent_config
apply_inference_overrides = _BASE.apply_inference_overrides
configure_robot_and_simulator = _BASE.configure_robot_and_simulator


def make_env_config(robot_cfg, args, stage):
    cfg = _BASE.env_config(robot_cfg, args)
    values = STAGES[stage]
    cfg.control_components["steering"] = ContinuousSteeringConfig(
        heading_change_steps_min=90,
        heading_change_steps_max=181,
        random_heading_probability=0.0,
        enable_rand_facing=False,
        yaw_rate=0.35,
        turn_angle_max=0.7853981633974483,
        **values,
    )
    return cfg
