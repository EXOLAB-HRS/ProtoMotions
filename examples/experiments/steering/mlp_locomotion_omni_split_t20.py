"""One-variable V2 ablation with a stricter travel-direction channel.

Everything is inherited from ``mlp_locomotion_omni_split_s20.py`` except the
tangent-velocity error scale, which changes from 0.025 to 2.0.  This candidate
is intentionally not launched until the V2 dense sweep proves that direction
tracking, rather than speed or stability, is the next limiting factor.
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
    cfg.reward_components["heading_rew"].static_params["tangent_err_scale"] = 2.0
    return cfg
