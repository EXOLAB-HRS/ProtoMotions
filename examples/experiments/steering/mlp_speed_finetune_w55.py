"""Phase-2 speed curriculum: make speed accuracy dominate motion reward.

Identical to mlp_speed_finetune.py except weights are speed=0.55,
direction=0.15, facing=0.30.  Phase 1's equal speed/direction split allowed a
slow but correctly directed gait to retain too much reward.
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
    params = cfg.reward_components["heading_rew"].static_params
    params["speed_reward_w"] = 0.55
    params["direction_reward_w"] = 0.15
    return cfg
