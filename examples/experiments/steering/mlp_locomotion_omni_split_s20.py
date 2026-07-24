"""Omnidirectional locomotion fine-tune with decoupled speed/direction reward.

One conceptual change from omni v1: replace the coupled heading kernel with the
split kernel.  Task/style weights, data, commands, optimizer, and network remain
unchanged.  Speed sharpness 2.0 is the best prior continuous-speed setting.
"""

import importlib.util
from pathlib import Path

from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.rewards import compute_split_heading_velocity_rew

_BASE_PATH = Path(__file__).with_name("mlp_locomotion_omni_v1.py")
_SPEC = importlib.util.spec_from_file_location("locomotion_omni_v1", _BASE_PATH)
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

terrain_config = _BASE.terrain_config
scene_lib_config = _BASE.scene_lib_config
motion_lib_config = _BASE.motion_lib_config
agent_config = _BASE.agent_config
apply_inference_overrides = _BASE.apply_inference_overrides


def env_config(robot_cfg, args):
    cfg = _BASE.env_config(robot_cfg, args)
    cfg.reward_components["heading_rew"] = MdpComponent(
        compute_func=compute_split_heading_velocity_rew,
        dynamic_vars={
            "root_pos": EnvContext.current.root_pos,
            "prev_root_pos": EnvContext.steering.prev_root_pos,
            "root_rot": EnvContext.current.root_rot,
            "tar_dir": EnvContext.steering.tar_dir,
            "tar_speed": EnvContext.steering.tar_speed,
            "tar_face_dir": EnvContext.steering.tar_face_dir,
            "dt": EnvContext.dt,
        },
        static_params={
            "weight": 1.0,
            "speed_err_scale": 2.0,
            "tangent_err_scale": 0.025,
            "speed_reward_w": 0.35,
            "direction_reward_w": 0.35,
            "facing_reward_w": 0.30,
        },
    )
    return cfg
