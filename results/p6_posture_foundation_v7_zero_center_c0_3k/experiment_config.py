"""Developmental ⑥ curriculum: learn balance before block ④ exists.

The actor sees only seven sensory posture values and emits a bounded correction
around the neutral PD target.  Static pose snapshots provide non-locomotor style
examples; no steering command, gait phase, previous action, or Motion Driver
checkpoint reaches the actor.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

from human_controller.posture import (
    compute_posture_observation,
    compute_posture_stability_reward,
)
from protomotions.envs.context_views import EnvContext
from protomotions.envs.control.steering_control import SteeringControlConfig
from protomotions.envs.mdp_component import MdpComponent
from protomotions.simulator.base_simulator.config import (
    DomainRandomizationConfig,
    PushDomainRandomizationConfig,
)

_BASE_PATH = (
    Path(__file__).resolve().parents[2]
    / "ProtoMotions"
    / "examples"
    / "experiments"
    / "steering"
    / "mlp.py"
)
_SPEC = importlib.util.spec_from_file_location("posture_foundation_base", _BASE_PATH)
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

scene_lib_config = _BASE.scene_lib_config
motion_lib_config = _BASE.motion_lib_config
apply_inference_overrides = _BASE.apply_inference_overrides
_NEUTRAL_POSTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "artifacts"
    / "posture_neutral_base_action_v2.json"
)


def _apply_neutral_posture_reset(robot_cfg) -> None:
    """Start every episode from the same pose the fixed base action holds."""
    with _NEUTRAL_POSTURE_PATH.open() as f:
        posture = json.load(f)
    dof_pos = posture["reset_dof_pos"]
    if len(dof_pos) != robot_cfg.default_dof_pos.numel():
        raise ValueError("neutral posture DOF count does not match the robot")
    robot_cfg.default_dof_pos[:] = robot_cfg.default_dof_pos.new_tensor(dof_pos)
    robot_cfg.default_root_height = float(posture["reset_root_height"])


def terrain_config(args):
    cfg = _BASE.terrain_config(args)
    # There are no scene objects in the posture curriculum.  Keeping the
    # upstream 10 m empty-scene spacing makes a huge flat mesh as num_envs
    # grows and adds many minutes to startup without changing the physics.
    cfg.spacing_between_scenes = 1.0
    return cfg


def env_config(robot_cfg, args):
    cfg = _BASE.env_config(robot_cfg, args)
    # The upstream normalized action uses the joint-limit midpoint as offset.
    # Recenter it on the reference stand, so action=0 holds the same pose used
    # at reset and the learned policy emits only posture residuals around it.
    cfg.action_config["pd_action_offset"] = robot_cfg.default_dof_pos.clone()
    cfg.control_components["steering"] = SteeringControlConfig(
        tar_speed_min=0.0,
        tar_speed_max=0.0,
        stop_probability=1.0,
        enable_rand_facing=False,
        heading_change_steps_min=100000,
        heading_change_steps_max=100001,
    )
    cfg.observation_components["posture_obs"] = MdpComponent(
        compute_func=compute_posture_observation,
        dynamic_vars={
            "root_rot": EnvContext.current.root_rot,
            "root_ang_vel": EnvContext.current.root_ang_vel,
            "root_pos": EnvContext.current.root_pos,
            "ground_height": EnvContext.ground_heights,
        },
    )
    cfg.reward_components = {
        "posture_stability": MdpComponent(
            compute_func=compute_posture_stability_reward,
            dynamic_vars={
                "root_rot": EnvContext.current.root_rot,
                "root_ang_vel": EnvContext.current.root_ang_vel,
                "root_pos": EnvContext.current.root_pos,
                "root_vel": EnvContext.current.root_vel,
                "ground_height": EnvContext.ground_heights,
            },
            static_params={
                "target_height": float(
                    os.environ.get("HC_POSTURE_TARGET_HEIGHT", "0.93")
                ),
                "weight": 1.0,
            },
        )
    }
    cfg.max_episode_length = 240
    return cfg


def configure_robot_and_simulator(robot_cfg, simulator_cfg, args):
    _apply_neutral_posture_reset(robot_cfg)
    stage = int(os.environ.get("HC_POSTURE_CURRICULUM_STAGE", "0"))
    settings = {
        0: ((3.0, 5.0), (0.05, 0.05, 0.0), (0.04, 0.04, 0.02)),
        1: ((2.0, 3.5), (0.20, 0.20, 0.0), (0.15, 0.15, 0.08)),
        2: ((1.25, 2.5), (0.35, 0.35, 0.0), (0.30, 0.30, 0.15)),
    }
    if stage not in settings:
        raise ValueError("HC_POSTURE_CURRICULUM_STAGE must be 0, 1, or 2")
    interval, linear, angular = settings[stage]
    simulator_cfg.domain_randomization = DomainRandomizationConfig(
        push=PushDomainRandomizationConfig(
            push_interval_range=interval,
            max_linear_velocity=linear,
            max_angular_velocity=angular,
        )
    )


def agent_config(robot_cfg, env_cfg, args):
    cfg = _BASE.agent_config(robot_cfg, env_cfg, args)
    if "posture_obs" not in cfg.model.in_keys:
        cfg.model.in_keys.append("posture_obs")
    cfg.model.actor.in_keys = ["posture_obs"]
    cfg.model.actor.mu_model.in_keys = ["posture_obs"]
    cfg.model.actor.mu_model._target_ = (
        "human_controller.models.posture_stabilized_motion_driver."
        "PostureFoundationPolicy"
    )
    cfg.model.actor.actor_logstd = float(
        os.environ.get("HC_POSTURE_ACTOR_LOGSTD", "-3.5")
    )
    cfg.model.actor_optimizer.lr = float(
        os.environ.get("HC_POSTURE_ACTOR_LR", "5e-5")
    )
    cfg.task_reward_w = 0.8
    cfg.amp_parameters.discriminator_reward_threshold = 0.0
    cfg.amp_parameters.discriminator_reward_w = 0.2
    return cfg
