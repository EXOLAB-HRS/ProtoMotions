"""Locomotion-only omnidirectional steering baseline.

This is a fresh AMP run (not a warm start from the mixed ACCAD policy).  The
motion file is supplied by the launcher and is expected to be the curated,
class-balanced locomotion library.
"""

import importlib.util
from pathlib import Path

_BASE_PATH = Path(__file__).with_name("mlp.py")
_SPEC = importlib.util.spec_from_file_location("steering_mlp_base", _BASE_PATH)
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

scene_lib_config = _BASE.scene_lib_config
motion_lib_config = _BASE.motion_lib_config
agent_config = _BASE.agent_config
apply_inference_overrides = _BASE.apply_inference_overrides


def terrain_config(args):
    cfg = _BASE.terrain_config(args)
    cfg.spacing_between_scenes = 1.0
    return cfg


def env_config(robot_cfg, args):
    cfg = _BASE.env_config(robot_cfg, args)
    steering = cfg.control_components["steering"]
    steering.tar_speed_min = 0.5
    steering.tar_speed_max = 1.5
    steering.heading_change_steps_min = 150
    steering.heading_change_steps_max = 301
    steering.random_heading_probability = 0.2
    steering.standard_heading_change = 0.5
    steering.standard_speed_change = 0.35
    steering.stop_probability = 0.0
    # Independent travel/facing is what distinguishes genuine lateral and
    # backward walking from a rotated forward walk.
    steering.enable_rand_facing = True
    return cfg
