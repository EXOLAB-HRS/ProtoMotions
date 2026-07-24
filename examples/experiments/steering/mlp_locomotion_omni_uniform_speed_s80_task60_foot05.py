"""V9: V7 with a midpoint task/style balance.

This is a one-variable ablation of V7: change only ``task_reward_w`` from
0.70 to 0.60.  V7 established continuous command tracking but its lateral
motion visibly skated; V8 showed that doubling foot-plant credit worsened
sliding and initialization robustness.  This treatment instead restores more
AMP walking-style pressure while retaining V7's successful foot reward.
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
env_config = _BASE.env_config
apply_inference_overrides = _BASE.apply_inference_overrides


def agent_config(robot_cfg, env_cfg, args):
    cfg = _BASE.agent_config(robot_cfg, env_cfg, args)
    cfg.task_reward_w = 0.60
    return cfg
