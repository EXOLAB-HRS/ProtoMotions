"""Backward MC fine-tune driven only by command-task credit.

The parent keeps the frozen BS and deploy-time Composer in the rollout.  This
variant removes AMP advantage and AMP-based termination only for the short
backward specialization stage; the actor remains warm-started from the
style-trained balanced checkpoint.
"""

import importlib.util
from pathlib import Path


_BASE_PATH = Path(__file__).with_name(
    "mlp_locomotion_on_frozen_posture_backward_focus.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "locomotion_on_frozen_posture_backward_focus", _BASE_PATH
)
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

terrain_config = _BASE.terrain_config
scene_lib_config = _BASE.scene_lib_config
motion_lib_config = _BASE.motion_lib_config
env_config = _BASE.env_config
apply_inference_overrides = _BASE.apply_inference_overrides
configure_robot_and_simulator = _BASE.configure_robot_and_simulator


def agent_config(robot_cfg, env_cfg, args):
    cfg = _BASE.agent_config(robot_cfg, env_cfg, args)
    cfg.task_reward_w = 1.0
    cfg.amp_parameters.discriminator_reward_w = 0.0
    cfg.amp_parameters.discriminator_max_cumulative_bad_transitions = 1000000
    return cfg
