"""Curriculum fine-tune config for continuous speed tracking.

Warm-start this config from the stable full2 epoch-2000 MLP.  Phase 1 isolates
speed calibration: moderate speed sharpness, long-lived commands, no stop targets,
and facing aligned with travel.  Direction and speed penalties are independent.
"""

import importlib.util
from pathlib import Path

from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.rewards import compute_split_heading_velocity_rew

_BASE_PATH = Path(__file__).with_name("mlp.py")
_SPEC = importlib.util.spec_from_file_location("steering_mlp_base", _BASE_PATH)
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

scene_lib_config = _BASE.scene_lib_config
motion_lib_config = _BASE.motion_lib_config
agent_config = _BASE.agent_config
apply_inference_overrides = _BASE.apply_inference_overrides


def terrain_config(args):
    cfg = _BASE.terrain_config(args)
    # No scene objects are used in steering. The upstream Terrain nevertheless
    # reserves one empty scene slot per env; at 4096 envs and 10 m spacing this
    # inflates a flat heightfield to ~66M vertices and costs ~20 minutes/start.
    # Smaller empty-slot spacing leaves the actual flat locomotion terrain and
    # physics unchanged while reducing initialization substantially.
    cfg.spacing_between_scenes = 1.0
    return cfg


def env_config(robot_cfg, args):
    cfg = _BASE.env_config(robot_cfg, args)

    steering = cfg.control_components["steering"]
    steering.tar_speed_min = 0.5
    steering.tar_speed_max = 1.5
    steering.heading_change_steps_min = 300
    steering.heading_change_steps_max = 301
    steering.stop_probability = 0.0
    steering.enable_rand_facing = False

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
            "speed_err_scale": 0.5,
            "tangent_err_scale": 0.025,
            "speed_reward_w": 0.35,
            "direction_reward_w": 0.35,
            "facing_reward_w": 0.30,
        },
    )
    return cfg
