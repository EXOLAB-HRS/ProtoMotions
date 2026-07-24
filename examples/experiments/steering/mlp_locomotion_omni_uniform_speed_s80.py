"""V5: V4 uniform command sampling with a sharper speed-error kernel.

This is a one-variable ablation of ``mlp_locomotion_omni_uniform_speed``:
``speed_err_scale`` changes from 2.0 to 8.0.  Data, command sampling,
speed/direction/facing channel weights, task/style weights, network, and
optimizer remain unchanged.
"""

import importlib.util
from pathlib import Path

_BASE_PATH = Path(__file__).with_name("mlp_locomotion_omni_uniform_speed.py")
_SPEC = importlib.util.spec_from_file_location("locomotion_omni_uniform_speed", _BASE_PATH)
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

terrain_config = _BASE.terrain_config
scene_lib_config = _BASE.scene_lib_config
motion_lib_config = _BASE.motion_lib_config
agent_config = _BASE.agent_config
apply_inference_overrides = _BASE.apply_inference_overrides


def env_config(robot_cfg, args):
    cfg = _BASE.env_config(robot_cfg, args)
    cfg.reward_components["heading_rew"].static_params["speed_err_scale"] = 8.0
    return cfg
