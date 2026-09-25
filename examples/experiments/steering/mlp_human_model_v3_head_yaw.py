"""a12 head envelope + head-chest yaw rate term (second fine-tune, f2 stage).

Used by p2f2 (and p1f2 on the local PC) after the f1 stage (facing = travel
direction, 0-1.5 m/s with stops, a12 head term). Command settings are launcher
overrides, not set here. The two head terms share a12's task share (0.15)
equally. Settings table: share/training_configs/p2_series_server/README.md.
"""

import importlib.util
import math
from pathlib import Path

from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.rewards.locomotion_quality import head_yaw_stability


_SPEC = importlib.util.spec_from_file_location(
    "head_yaw_a12_base", Path(__file__).with_name("mlp_human_model_v3_a12_scratch_head.py")
)
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

EXPERIMENT_ID = "head_yaw"
HEAD_ENVELOPE_WEIGHT = 0.075
HEAD_YAW_WEIGHT = 0.075
# 2 x reference max clip-mean |w_head,z - w_chest,z| (42.9 deg/s), the same
# rule as the a7-headroll roll-rate scale.
HEAD_YAW_RATE_SCALE_RAD_S = math.radians(85.8)

terrain_config = _BASE.terrain_config
scene_lib_config = _BASE.scene_lib_config
motion_lib_config = _BASE.motion_lib_config
agent_config = _BASE.agent_config
apply_inference_overrides = _BASE.apply_inference_overrides
configure_robot_and_simulator = _BASE.configure_robot_and_simulator


def env_config(robot_cfg, args):
    cfg = _BASE.env_config(robot_cfg, args)
    names = robot_cfg.kinematic_info.body_names
    heading = cfg.reward_components["heading_rew"].static_params
    envelope = cfg.reward_components["reference_head_motion"].static_params
    assert abs(heading["weight"] + envelope["weight"] - 1.0) < 1e-9
    envelope["weight"] = HEAD_ENVELOPE_WEIGHT
    cfg.reward_components["head_yaw_stability"] = MdpComponent(
        compute_func=head_yaw_stability,
        dynamic_vars={"body_ang_vel": EnvContext.current.rigid_body_ang_vel},
        static_params={
            "head_index": names.index("Head"),
            "chest_index": names.index("Chest"),
            "yaw_rate_scale_rad_s": HEAD_YAW_RATE_SCALE_RAD_S,
            "weight": HEAD_YAW_WEIGHT,
        },
    )
    return cfg
