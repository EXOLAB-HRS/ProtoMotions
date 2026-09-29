"""Frozen-BS MC fine-tune with unbiased omnidirectional commands.

The parent V11-derived task samples a fully random heading only 20% of the
time; the other 80% is a small change from the previous heading.  Because the
initial heading is forward, short fine-tunes overrepresent forward locomotion.
This ablation changes only heading sampling: every task reset draws a uniform
heading over [-pi, pi].  Speed was already sampled uniformly and remains so.
"""

import importlib.util
import os
from pathlib import Path

from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import MdpComponent
from human_controller.posture_free import copy_previous_action


_BASE_PATH = Path(__file__).with_name("mlp_locomotion_on_frozen_posture.py")
_SPEC = importlib.util.spec_from_file_location(
    "locomotion_on_frozen_posture", _BASE_PATH
)
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

terrain_config = _BASE.terrain_config
scene_lib_config = _BASE.scene_lib_config
motion_lib_config = _BASE.motion_lib_config
agent_config = _BASE.agent_config
apply_inference_overrides = _BASE.apply_inference_overrides
configure_robot_and_simulator = _BASE.configure_robot_and_simulator


def env_config(robot_cfg, args):
    cfg = _BASE.env_config(robot_cfg, args)
    steering = cfg.control_components["steering"]
    steering.random_heading_probability = 1.0
    steering.random_speed_probability = 1.0
    if float(os.environ.get("HC_TRAINING_COMPOSER_ALPHA", "1.0")) < 1.0:
        cfg.observation_components["composer_prev_action"] = MdpComponent(
            compute_func=copy_previous_action,
            dynamic_vars={"previous_action": EnvContext.previous_action},
        )
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    cfg = _BASE.agent_config(robot_cfg, env_cfg, args)
    if float(os.environ.get("HC_TRAINING_COMPOSER_ALPHA", "1.0")) < 1.0:
        if "composer_prev_action" not in cfg.model.in_keys:
            cfg.model.in_keys.append("composer_prev_action")
        if "composer_prev_action" not in cfg.model.actor.in_keys:
            cfg.model.actor.in_keys.append("composer_prev_action")
    return cfg
