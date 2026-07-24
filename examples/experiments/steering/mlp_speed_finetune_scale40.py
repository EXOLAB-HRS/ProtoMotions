"""Continuous-speed fine-tune with speed error scale 4.0.

One-variable follow-up to ``mlp_speed_finetune_scale20.py``.  All curriculum,
reward weights, and training settings remain unchanged; only the speed-error
scale changes from 2.0 to 4.0.
"""

import importlib.util
from pathlib import Path

_BASE_PATH = Path(__file__).with_name("mlp_speed_finetune_scale20.py")
_SPEC = importlib.util.spec_from_file_location("steering_speed_scale20_base", _BASE_PATH)
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

terrain_config = _BASE.terrain_config
scene_lib_config = _BASE.scene_lib_config
motion_lib_config = _BASE.motion_lib_config
agent_config = _BASE.agent_config
apply_inference_overrides = _BASE.apply_inference_overrides


def env_config(robot_cfg, args):
    cfg = _BASE.env_config(robot_cfg, args)
    cfg.reward_components["heading_rew"].static_params["speed_err_scale"] = 4.0
    return cfg
