"""V7: V6 plus a small contact-phase foot-plant reward.

This is a one-variable ablation of V6: add ``foot_plant_rew`` with weight 0.05.
All steering, AMP, sampling, architecture, and optimizer settings remain fixed.
The experiment uses flat terrain, so a soft foot-height mask supplies the
contact phase without adding expensive simulator contact sensors.
"""

import importlib.util
from pathlib import Path

import torch

from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.rewards.task import compute_foot_plant_rew

_BASE_PATH = Path(__file__).with_name(
    "mlp_locomotion_omni_uniform_speed_s80_task70.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "locomotion_omni_uniform_speed_s80_task70", _BASE_PATH
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
    foot_names = (
        robot_cfg.common_naming_to_robot_body_names["all_left_foot_bodies"]
        + robot_cfg.common_naming_to_robot_body_names["all_right_foot_bodies"]
    )
    body_names = robot_cfg.kinematic_info.body_names
    foot_body_ids = torch.tensor(
        [body_names.index(name) for name in foot_names], dtype=torch.long
    )
    cfg.reward_components["foot_plant_rew"] = MdpComponent(
        compute_func=compute_foot_plant_rew,
        dynamic_vars={
            "rigid_body_pos": EnvContext.current.rigid_body_pos,
            "rigid_body_vel": EnvContext.current.rigid_body_vel,
        },
        static_params={
            "foot_body_ids": foot_body_ids,
            "contact_height": 0.12,
            "contact_margin": 0.04,
            "velocity_scale": 4.0,
            "weight": 0.05,
        },
    )
    return cfg
