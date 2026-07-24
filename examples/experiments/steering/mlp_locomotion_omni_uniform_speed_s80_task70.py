"""V6: V5 with stronger steering-task credit.

This is a one-variable ablation of ``mlp_locomotion_omni_uniform_speed_s80``:
``task_reward_w`` changes from 0.5 to 0.7.  The discriminator/style weight,
motion data, uniform command sampling, reward kernel/channel weights, network,
and optimizer remain unchanged.
"""

import importlib.util
from pathlib import Path

_BASE_PATH = Path(__file__).with_name("mlp_locomotion_omni_uniform_speed_s80.py")
_SPEC = importlib.util.spec_from_file_location("locomotion_omni_uniform_speed_s80", _BASE_PATH)
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

terrain_config = _BASE.terrain_config
scene_lib_config = _BASE.scene_lib_config
motion_lib_config = _BASE.motion_lib_config
env_config = _BASE.env_config
apply_inference_overrides = _BASE.apply_inference_overrides


def agent_config(robot_cfg, env_cfg, args):
    cfg = _BASE.agent_config(robot_cfg, env_cfg, args)
    cfg.task_reward_w = 0.7
    return cfg
