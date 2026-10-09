"""teacher_cadence_v1 revision 25: r22 with one change, heading_rew vel_err_scale 8 -> VEL_ERR_SCALE.

r22 (best leg kinematics: knee at contact 10.0 deg, Lencioni knee r 0.955, ankle r 0.899, 1.78 steps/s) walked
1.18 m/s at a 1 m/s command. Steady-state speed is scored only by heading_rew (speed_transition_tracking is 1
outside transitions); at scale 8 a 0.18 m/s excess keeps exp(-8 * 0.18^2) = 0.77 of the direction term, at 20 0.52.
r24 changed three terms at once and regressed (ankle r 0.81, 1.94 steps/s), so this run changes one.
Starts from r22 last.ckpt.
"""
from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r22 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r22 import (
    env_config as _parent_env_config, agent_config)  # noqa: F401

VEL_ERR_SCALE = 20.0


def env_config(robot_cfg, args):
    cfg = _parent_env_config(robot_cfg, args)
    cfg.reward_components["heading_rew"].static_params["vel_err_scale"] = VEL_ERR_SCALE
    return cfg
