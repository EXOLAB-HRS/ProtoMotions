"""Scale-2.0 continuous-speed fine-tune with 1% direct upright reward.

One-variable interpolation between Phase 3 (0%) and the over-strong Phase 5
(5%).  All other reward, curriculum, and training settings remain fixed.
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
    cfg.reward_components["heading_rew"].static_params["upright_reward_w"] = 0.01
    return cfg
