"""teacher_cadence_v1 revision 22: r21 (role split: AMASS natural AMP pack, AMP weight 0.4; Lencioni Hip/Knee/Ankle
leg reward) with knee_strike_prior weight 0.30 -> 0.60 (r19 showed 22.5 -> 15.6 deg knee at contact).
Starts from r21 last.ckpt.
"""
from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r21 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r21 import (
    env_config as _parent_env_config, agent_config)  # noqa: F401

KNEE_STRIKE_WEIGHT = 0.60


def env_config(robot_cfg, args):
    cfg = _parent_env_config(robot_cfg, args)
    cfg.reward_components["knee_strike_prior"].static_params["weight"] = KNEE_STRIKE_WEIGHT
    return cfg
