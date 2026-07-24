"""V8: V7 with stronger contact-phase foot-plant credit.

This is a one-variable ablation of V7: change only ``foot_plant_rew`` weight
from 0.05 to 0.10.  The purpose is to turn the quantitatively improved but
visually sliding strafe into an actual alternating lateral step pattern.
"""

import importlib.util
from pathlib import Path


_BASE_PATH = Path(__file__).with_name(
    "mlp_locomotion_omni_uniform_speed_s80_task70_foot05.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "locomotion_omni_uniform_speed_s80_task70_foot05", _BASE_PATH
)
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

terrain_config = _BASE.terrain_config
scene_lib_config = _BASE.scene_lib_config
motion_lib_config = _BASE.motion_lib_config
agent_config = _BASE.agent_config
apply_inference_overrides = _BASE.apply_inference_overrides


def env_config(robot_cfg, args):
    cfg = _BASE.env_config(robot_cfg, args)
    cfg.reward_components["foot_plant_rew"].static_params["weight"] = 0.10
    return cfg
