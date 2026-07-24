"""Stage-0 locomotion curriculum: learn control without AMP early termination.

The discriminator and its style reward remain fully active.  The only change
from omni v1 is a zero discriminator-reward termination threshold, preventing a
fresh policy from being reset after ten consecutive low-style transitions.
"""

import importlib.util
from pathlib import Path

_BASE_PATH = Path(__file__).with_name("mlp_locomotion_omni_v1.py")
_SPEC = importlib.util.spec_from_file_location("locomotion_omni_v1", _BASE_PATH)
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

terrain_config = _BASE.terrain_config
scene_lib_config = _BASE.scene_lib_config
motion_lib_config = _BASE.motion_lib_config
env_config = _BASE.env_config
apply_inference_overrides = _BASE.apply_inference_overrides


def agent_config(robot_config, env_config, args):
    cfg = _BASE.agent_config(robot_config, env_config, args)
    cfg.amp_parameters.discriminator_reward_threshold = 0.0
    return cfg
