"""V4: uniformly sample the full speed range independently of heading.

This branches from V2, the best moving omnidirectional parent.  The sole
conceptual change is command sampling: every speed reset is uniform over
0.5--1.5 m/s instead of being coupled to the 0.2 random-heading branch.
Reward shape/weights, data, network, optimizer, and direction/facing sampling
are otherwise identical to V2.
"""

import importlib.util
from pathlib import Path

_BASE_PATH = Path(__file__).with_name("mlp_locomotion_omni_split_s20.py")
_SPEC = importlib.util.spec_from_file_location("locomotion_omni_split_s20", _BASE_PATH)
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

terrain_config = _BASE.terrain_config
scene_lib_config = _BASE.scene_lib_config
motion_lib_config = _BASE.motion_lib_config
agent_config = _BASE.agent_config
apply_inference_overrides = _BASE.apply_inference_overrides


def env_config(robot_cfg, args):
    cfg = _BASE.env_config(robot_cfg, args)
    cfg.control_components["steering"].random_speed_probability = 1.0
    return cfg
