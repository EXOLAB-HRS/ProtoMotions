"""Continuous-speed fine-tune with a sharper speed-only reward.

One-variable ablation of ``mlp_speed_finetune.py``: speed error scale changes
from 0.5 to 2.0.  The split kernel keeps tangent/direction sharpness fixed,
unlike the earlier coupled-reward experiment.
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
agent_config = _BASE.agent_config
apply_inference_overrides = _BASE.apply_inference_overrides


def env_config(robot_cfg, args):
    cfg = _BASE.env_config(robot_cfg, args)
    cfg.reward_components["heading_rew"].static_params["speed_err_scale"] = 2.0
    return cfg
