"""a12: a11 scratch omni training plus reference-bounded head stability."""

import importlib.util
import math
from pathlib import Path

from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.rewards.locomotion_quality import reference_bounded_head_motion


_SPEC = importlib.util.spec_from_file_location(
    "a12_base", Path(__file__).with_name("mlp_human_model_v3_a11_proto1.py")
)
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

EXPERIMENT_ID = "a12"
HEAD_REWARD_WEIGHT = 0.15
HEAD_TILT_LIMIT_RAD = math.radians(15.7145)
HEAD_RELATIVE_SPEED_LIMIT_RAD_S = math.radians(77.2072)

terrain_config = _BASE.terrain_config
scene_lib_config = _BASE.scene_lib_config
motion_lib_config = _BASE.motion_lib_config
agent_config = _BASE.agent_config
apply_inference_overrides = _BASE.apply_inference_overrides
configure_robot_and_simulator = _BASE.configure_robot_and_simulator


def env_config(robot_cfg, args):
    cfg = _BASE.env_config(robot_cfg, args)
    names = robot_cfg.kinematic_info.body_names
    cfg.reward_components["heading_rew"].static_params["weight"] = 1 - HEAD_REWARD_WEIGHT
    cfg.reward_components["reference_head_motion"] = MdpComponent(
        compute_func=reference_bounded_head_motion,
        dynamic_vars={
            "body_rot": EnvContext.current.rigid_body_rot,
            "body_ang_vel": EnvContext.current.rigid_body_ang_vel,
        },
        static_params={
            "head_index": names.index("Head"),
            "chest_index": names.index("Chest"),
            "tilt_limit_rad": HEAD_TILT_LIMIT_RAD,
            "relative_speed_limit_rad_s": HEAD_RELATIVE_SPEED_LIMIT_RAD_S,
            "weight": HEAD_REWARD_WEIGHT,
        },
    )
    return cfg
