"""Sensory-only BS fine-tune on rear-sector locomotion with push exposure."""

import importlib.util
from pathlib import Path

import numpy as np


_BASE_PATH = Path(__file__).with_name(
    "mlp_locomotion_free_posture_residual.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "locomotion_free_posture_residual", _BASE_PATH
)
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

terrain_config = _BASE.terrain_config
scene_lib_config = _BASE.scene_lib_config
motion_lib_config = _BASE.motion_lib_config
agent_config = _BASE.agent_config
apply_inference_overrides = _BASE.apply_inference_overrides
configure_robot_and_simulator = _BASE.configure_robot_and_simulator


def env_config(robot_cfg, args):
    cfg = _BASE.env_config(robot_cfg, args)
    steering = cfg.control_components["steering"]
    steering.random_heading_probability = 1.0
    steering.random_speed_probability = 1.0
    steering.random_heading_min = 0.75 * np.pi
    steering.random_heading_max = 1.25 * np.pi
    return cfg
