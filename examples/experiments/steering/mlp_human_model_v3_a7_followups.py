"""Shared candidate configuration for a9, a10, b1-a7; no new model version."""
import importlib.util
from pathlib import Path
from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.rewards.locomotion_quality import (
    head_upright, head_angular_stability, target_second_difference,
)
from protomotions.envs.control.continuous_steering import ContinuousSteeringConfig

_spec = importlib.util.spec_from_file_location("a7_followup_base", Path(__file__).with_name("mlp_human_model_v3_stage_a.py"))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
terrain_config = _base.terrain_config
scene_lib_config = _base.scene_lib_config
motion_lib_config = _base.motion_lib_config
configure_robot_and_simulator = _base.configure_robot_and_simulator
apply_inference_overrides = _base.apply_inference_overrides


def discriminator_batch_size(ppo_batch_size):
    """Keep AMP batches valid when local smoke/VRAM overrides shrink PPO."""
    return min(4096, ppo_batch_size)


def agent_config(robot_cfg, env_cfg, args):
    cfg = _base.agent_config(robot_cfg, env_cfg, args)
    cfg.task_reward_w = 0.6
    cfg.amp_parameters.discriminator_reward_w = 0.4
    cfg.model.discriminator_optimizer.lr = 1e-4
    cfg.model.actor_optimizer.lr = 2e-5
    cfg.model.critic_optimizer.lr = 1e-4
    cfg.model.disc_critic_optimizer.lr = 1e-4
    cfg.amp_parameters.discriminator_batch_size = discriminator_batch_size(
        args.batch_size
    )
    cfg.save_epoch_checkpoint_every = max(1, round(5_000_000 / (args.num_envs * 32)))
    return cfg


def make_env_config(robot_cfg, args, experiment):
    cfg = _base.env_config(robot_cfg, args)
    cfg.reward_components["heading_rew"].static_params["vel_err_scale"] = 2.
    if experiment == "a9":
        cfg.reward_components["heading_rew"].static_params["weight"] = .95
        head = robot_cfg.kinematic_info.body_names.index("Head")
        cfg.reward_components["head_upright"] = MdpComponent(
            compute_func=head_upright,
            dynamic_vars={"body_rot": EnvContext.current.rigid_body_rot},
            static_params={"body_index": head, "tilt_scale": .174533, "weight": .03})
        cfg.reward_components["head_angular_stability"] = MdpComponent(
            compute_func=head_angular_stability,
            dynamic_vars={"body_ang_vel": EnvContext.current.rigid_body_ang_vel},
            static_params={"body_index": head, "speed_scale": .8, "weight": .02})
    elif experiment == "a10":
        cfg.reward_components["heading_rew"].static_params["weight"] = .97
        cfg.reward_components["target_second_difference"] = MdpComponent(
            compute_func=target_second_difference,
            dynamic_vars={"current": EnvContext.current_processed_action,
                          "history": EnvContext.historical.processed_actions},
            static_params={"weight": .03, "scale": .05, "zero_during_grace_period": True})
    elif experiment == "b1-a7":
        cfg.control_components["steering"] = ContinuousSteeringConfig(
            tar_speed_min=.5, tar_speed_max=1.5,
            heading_change_steps_min=90, heading_change_steps_max=181,
            random_heading_probability=0., enable_rand_facing=False)
    else:
        raise ValueError(experiment)
    return cfg
