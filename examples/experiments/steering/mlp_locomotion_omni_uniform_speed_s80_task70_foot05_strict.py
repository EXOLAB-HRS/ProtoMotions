"""V10: V7 with worst-contact-body rather than mean foot-plant credit.

This changes one reward design variable: the foot-plant aggregation is
``strict_min`` instead of ``mean``.  Weight, contact-height mask, velocity
scale, task/style balance, command sampling, data, and optimizer stay at V7.
The treatment prevents one stationary ankle/toe from averaging away another
near-ground body that is visibly skating.
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
    cfg.reward_components["foot_plant_rew"].static_params[
        "aggregation"
    ] = "strict_min"
    return cfg
