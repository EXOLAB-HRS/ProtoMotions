"""Second-stage MC fine-tune focused on backward travel commands.

This inherits the balanced, frozen-BS, Composer-aware task and changes only
the fully random movement-heading interval to the rear 90-degree sector.
Facing remains independently random and speed remains uniform over V11's
0.5--1.5 m/s range.
"""

import importlib.util
import numpy as np
from pathlib import Path


_BASE_PATH = Path(__file__).with_name(
    "mlp_locomotion_on_frozen_posture_cardinal_balanced.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "locomotion_on_frozen_posture_cardinal_balanced", _BASE_PATH
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
    steering.random_heading_min = 0.75 * np.pi
    steering.random_heading_max = 1.25 * np.pi
    return cfg
