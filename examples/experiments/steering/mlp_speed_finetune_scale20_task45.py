"""Scale-2.0 continuous-speed fine-tune with slightly more AMP weight.

One-variable follow-up to ``mlp_speed_finetune_scale20.py``.  The dense sweep
improved speed MAE but missed the upright gate by 0.003, so only task_reward_w
changes from 0.50 to 0.45; all reward components and curriculum stay fixed.
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
apply_inference_overrides = _BASE.apply_inference_overrides


def env_config(robot_cfg, args):
    return _BASE.env_config(robot_cfg, args)


def agent_config(robot_cfg, env_cfg, args):
    cfg = _BASE.agent_config(robot_cfg, env_cfg, args)
    cfg.task_reward_w = 0.45
    return cfg
