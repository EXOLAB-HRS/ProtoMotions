"""A6 neck-reward v1: A5 plus a reference-bounded neck/head speed reward."""

import importlib.util
from pathlib import Path

from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.rewards.task import compute_neck_stability_rew


_SPEC = importlib.util.spec_from_file_location(
    "a6_base", Path(__file__).with_name("mlp_human_model_v3_stage_a_a5.py")
)
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

terrain_config = _BASE.terrain_config
scene_lib_config = _BASE.scene_lib_config
motion_lib_config = _BASE.motion_lib_config
agent_config = _BASE.agent_config
apply_inference_overrides = _BASE.apply_inference_overrides
configure_robot_and_simulator = _BASE.configure_robot_and_simulator

# Absolute joint-velocity p95 [rad/s], measured independently per axis from the
# exact A5/A6 v4 forward25 pack (SHA-256 6cf38a7d...9f43b84, 4,119 frames).
_REFERENCE_ABS_VELOCITY_P95 = (
    0.7622655034,
    1.4256707430,
    0.8655657172,
    0.4837467074,
    0.4921766818,
    0.4224115610,
)


def env_config(robot_cfg, args):
    cfg = _BASE.env_config(robot_cfg, args)
    names = robot_cfg.kinematic_info.dof_names
    neck_head_names = tuple(
        f"{joint}_{axis}" for joint in ("Neck", "Head") for axis in "xyz"
    )
    neck_head_ids = [names.index(name) for name in neck_head_names]

    # Preserve unit task-reward scale: 90% original steering + 10% neck quality.
    # The 10% share is an experiment-design choice, not an anatomical constant.
    cfg.reward_components["heading_rew"].static_params["weight"] = 0.9
    cfg.reward_components["neck_stability_rew"] = MdpComponent(
        compute_func=compute_neck_stability_rew,
        dynamic_vars={"dof_vel": EnvContext.current.dof_vel},
        static_params={
            "dof_indices": neck_head_ids,
            "reference_abs_velocity_p95": _REFERENCE_ABS_VELOCITY_P95,
            "weight": 0.1,
        },
    )
    return cfg
