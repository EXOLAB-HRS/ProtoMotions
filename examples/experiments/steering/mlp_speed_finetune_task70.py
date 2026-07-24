"""Continuous-speed fine-tune with stronger task-vs-style weighting.

This is a one-variable ablation of ``mlp_speed_finetune.py``: the steering
task reward weight is increased from 0.5 to 0.7.  Environment curriculum,
split reward shape, and speed/direction/facing weights remain unchanged.
"""

import importlib.util
from pathlib import Path

_BASE_PATH = Path(__file__).with_name("mlp_speed_finetune.py")
_SPEC = importlib.util.spec_from_file_location("steering_speed_ft05_base", _BASE_PATH)
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

terrain_config = _BASE.terrain_config
scene_lib_config = _BASE.scene_lib_config
motion_lib_config = _BASE.motion_lib_config
env_config = _BASE.env_config
apply_inference_overrides = _BASE.apply_inference_overrides


def agent_config(robot_cfg, args):
    cfg = _BASE.agent_config(robot_cfg, args)
    cfg.task_reward_w = 0.7
    return cfg
