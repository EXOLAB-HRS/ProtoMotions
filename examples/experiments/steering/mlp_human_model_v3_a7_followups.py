"""Shared candidate configuration for a9, a10, b1-a7; no new model version."""
import importlib.util
import math
from pathlib import Path
from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.rewards.locomotion_quality import (
    head_upright, head_angular_stability, target_second_difference, head_roll_stability,
    pelvis_yaw_stability, speed_transition_tracking,
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
    elif experiment == "a7":
        # The a7 condition itself: Stage A commands and rewards unchanged. Candidates
        # that vary only an agent-side setting use this so the env stays the control.
        pass
    elif experiment == "b1-a7":
        cfg.control_components["steering"] = ContinuousSteeringConfig(
            tar_speed_min=.5, tar_speed_max=1.5,
            heading_change_steps_min=90, heading_change_steps_max=181,
            random_heading_probability=0., enable_rand_facing=False)
    else:
        raise ValueError(experiment)
    return cfg


# ---------------------------------------------------------------------------
# Method 1 (A -> B -> C -> D fine-tune), owner jinsu. Evidence:
# output/260924/e01_head_posture_diagnosis/.
HEAD_ROLL_WEIGHT = .15              # same task share as a12's head term
HEAD_ROLL_SCALE_DEG = 8.03          # reference max of per-clip mean head roll
HEAD_ROLL_RATE_SCALE_DEG_S = 41.5   # 2 x reference max per-clip roll rate (20.76);
                                    # a7 sits at 36-40, so the kernel keeps a
                                    # gradient there instead of saturating at 0

# a13 stage b: A replay 30%, 0.7-1.4 m/s rate-limited speed changes, no stop or turn.
STAGE_B_STEERING = dict(
    fixed_fraction=.30, turn_fraction=0., stop_probability=0.,
    tar_speed_min=.7, tar_speed_max=1.4, acceleration_min=.4, acceleration_max=.8,
    full_heading_for_turn=False, independent_facing_fraction=0.,
    heading_change_steps_min=90, heading_change_steps_max=181,
    random_heading_probability=0., enable_rand_facing=False,
)


def add_head_roll(cfg, robot_cfg):
    names = robot_cfg.kinematic_info.body_names
    cfg.reward_components["heading_rew"].static_params["weight"] = 1 - HEAD_ROLL_WEIGHT
    cfg.reward_components["head_roll_stability"] = MdpComponent(
        compute_func=head_roll_stability,
        dynamic_vars={"body_rot": EnvContext.current.rigid_body_rot,
                      "body_ang_vel": EnvContext.current.rigid_body_ang_vel},
        static_params={"head_index": names.index("Head"), "root_index": names.index("Pelvis"),
                       "roll_scale_rad": math.radians(HEAD_ROLL_SCALE_DEG),
                       "roll_rate_scale_rad_s": math.radians(HEAD_ROLL_RATE_SCALE_DEG_S),
                       "weight": HEAD_ROLL_WEIGHT})
    return cfg


def set_stage_b_steering(cfg):
    cfg.control_components["steering"] = ContinuousSteeringConfig(**STAGE_B_STEERING)
    return cfg


# The a7 heading kernel is exp(-2 * err^2): a 0.15 m/s speed error, the Stage A
# limit, costs 3% of the heading reward, less than a quarter of what the head
# roll term takes for a7's 1.2 m/s tilt. Every candidate that added a second
# objective to it lost speed calibration (b1-a7, b2-a7, b2-a7-headroll). V11 and
# a11-a13 use 8, where the same error costs 16.5%.
SHARP_VEL_ERR_SCALE = 8.


def sharpen_speed_tracking(cfg):
    cfg.reward_components["heading_rew"].static_params["vel_err_scale"] = SHARP_VEL_ERR_SCALE
    return cfg


# Stage B settle fails mostly on deceleration: 1.4 -> 1.0 m/s takes 3-6.5 s against
# the 1.18 s the reference's slowest speed change allows, while 1.0 -> 1.4 is 1.1-2 s.
# Train on command ramps twice as steep as the gate's 0.4/0.8 m/s^2.
FAST_ACCELERATION = dict(acceleration_min=.8, acceleration_max=1.6)


def set_stage_b_fast_acceleration(cfg):
    cfg.control_components["steering"] = ContinuousSteeringConfig(**{**STAGE_B_STEERING, **FAST_ACCELERATION})
    return cfg


# Stage B fails every candidate on 1.4 m/s pelvis yaw (8.4 deg, almost all stride
# sway) against the 7.6 deg reference maximum, which the facing term barely sees.
# Scale = that maximum, as the head roll scale is the reference roll maximum.
PELVIS_YAW_WEIGHT = .10
PELVIS_YAW_SCALE_DEG = 7.6
STEEP_ACCELERATION = dict(acceleration_min=1.6, acceleration_max=3.2)


def add_pelvis_yaw(cfg):
    heading = cfg.reward_components["heading_rew"].static_params
    heading["weight"] = heading["weight"] - PELVIS_YAW_WEIGHT
    cfg.reward_components["pelvis_yaw_stability"] = MdpComponent(
        compute_func=pelvis_yaw_stability,
        dynamic_vars={"root_rot": EnvContext.current.root_rot,
                      "tar_face_dir": EnvContext.steering.tar_face_dir},
        static_params={"yaw_scale_rad": math.radians(PELVIS_YAW_SCALE_DEG),
                       "weight": PELVIS_YAW_WEIGHT})
    return cfg


def set_stage_b_steep_acceleration(cfg):
    cfg.control_components["steering"] = ContinuousSteeringConfig(**{**STAGE_B_STEERING, **STEEP_ACCELERATION})
    return cfg


# e11/e13/e14 fail Stage B only on settle. Measured with the reference's own rule
# (monotonic run of the 1 s trend), the policy changes body speed at 0.10-0.23 m/s^2,
# slower than every reference speed change (min 0.303, p10 0.339); response latency
# alone does not explain it (output/260924/e14.../deliverables/settle_decomposition.json).
# Speed changes are a few seconds out of every 3-6 s command, so the heading kernel
# averages the lag away. This term scores speed only during a ramp and 1 s after it.
# Scale 20: an error at the 0.10 m/s settle band keeps 82%, a 0.2 m/s lag 45%.
TRANSITION_WEIGHT = .15
TRANSITION_VEL_ERR_SCALE = 20.


def add_speed_transition(cfg):
    heading = cfg.reward_components["heading_rew"].static_params
    heading["weight"] = heading["weight"] - TRANSITION_WEIGHT
    cfg.reward_components["speed_transition_tracking"] = MdpComponent(
        compute_func=speed_transition_tracking,
        dynamic_vars={"root_pos": EnvContext.current.root_pos,
                      "prev_root_pos": EnvContext.steering.prev_root_pos,
                      "tar_dir": EnvContext.steering.tar_dir,
                      "tar_speed": EnvContext.steering.tar_speed,
                      "speed_transition": EnvContext.steering.speed_transition,
                      "dt": EnvContext.dt},
        static_params={"vel_err_scale": TRANSITION_VEL_ERR_SCALE, "weight": TRANSITION_WEIGHT})
    return cfg
