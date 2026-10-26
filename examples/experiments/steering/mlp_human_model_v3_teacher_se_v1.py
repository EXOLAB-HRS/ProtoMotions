"""teacher_se_v1 (2026-10-24): fixed-PD baseline teacher with the SE_v1 MC-teacher inputs, for training from scratch.

Rewards, AMP and plant follow teacher_cadence_v1_r25 (task 0.6 / AMP 0.4 on the AMASS train100 pack, Lencioni
Hip/Knee/Ankle gait table, knee strike 0.6, heading vel_err_scale 20, head roll, cadence terms) with one change from
r28: human_gait_prior ratio_tol 0.05 -> 0.30 (swing knee z -3.6 -> -2.0 at 1 m/s, criteria v2).

Inputs added for the SE_v1 I/O map (html/SE_v1_261024_io_map_jinsu_hyojun.html, MC teacher row), true values,
actor and critic only (the discriminator compares body motion with reference frames that have no such signals):
- se_gravity: gravity direction in the pelvis frame (3);
- se_specific_force: head specific force R_head^T (a_head - g) in m/s^2 (3);
- terrain: 24 x 5 height points, pelvis z - ground z, heading frame, forward -0.3..2.0 m (0.1 m) x lateral
  -0.4..0.4 m (0.2 m) (120). Constant on flat ground; present so terrain training keeps the same input layout.
Command interface (steering preview 5), cadence ratio and gait clock are r25's. Output: 59 PD targets.

This module is the stage-2 (full command) recipe; stage 1 is mlp_human_model_v3_teacher_se_v1_s1.py.
"""
from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r25 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r25 import (
    env_config as _r25_env_config, agent_config as _r25_agent_config, terrain_config as _r25_terrain_config)
from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.obs.se_teacher import compute_head_specific_force_obs, compute_pelvis_gravity_obs

RATIO_TOL = 0.30
HEIGHT_GRID_X = (-0.3, 2.0, 24)
HEIGHT_GRID_Y = (-0.4, 0.4, 5)
HEAD_BODY = "Head"
SE_KEYS = ["se_gravity", "se_specific_force", "terrain"]


def terrain_config(args):
    cfg = _r25_terrain_config(args)
    cfg.height_grid_x, cfg.height_grid_y = HEIGHT_GRID_X, HEIGHT_GRID_Y
    cfg.__post_init__()
    return cfg


def env_config(robot_cfg, args):
    cfg = _r25_env_config(robot_cfg, args)
    cfg.reward_components["human_gait_prior"].static_params["ratio_tol"] = RATIO_TOL
    head = list(robot_cfg.kinematic_info.body_names).index(HEAD_BODY)
    cfg.observation_components["se_gravity"] = MdpComponent(
        compute_func=compute_pelvis_gravity_obs,
        dynamic_vars={"root_rot": EnvContext.current.root_rot},
        static_params={"w_last": True})
    cfg.observation_components["se_specific_force"] = MdpComponent(
        compute_func=compute_head_specific_force_obs,
        dynamic_vars={"body_rot": EnvContext.current.rigid_body_rot, "body_vel": EnvContext.current.rigid_body_vel,
                      "historical_body_vel": EnvContext.historical.rigid_body_vel, "dt": EnvContext.dt},
        static_params={"head_idx": head, "w_last": True})
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    cfg = _r25_agent_config(robot_cfg, env_cfg, args)
    m = cfg.model
    for module in (m, m.actor, m.actor.mu_model, m.critic):
        module.in_keys = list(module.in_keys) + SE_KEYS
    return cfg
