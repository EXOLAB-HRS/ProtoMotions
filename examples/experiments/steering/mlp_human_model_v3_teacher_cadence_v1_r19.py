"""teacher_cadence_v1 revision 19: r17 with knee_strike_prior weight 0.30 -> 0.60.

r17: knee at contact 22.5 deg (r16 25.7), Fukuchi knee r 0.95; the knee still extends late in terminal
swing. Starts from r17 last.ckpt.
"""
from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r17 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r17 import (
    env_config as _r17_env_config, agent_config)  # noqa: F401

KNEE_STRIKE_WEIGHT = 0.60


def env_config(robot_cfg, args):
    cfg = _r17_env_config(robot_cfg, args)
    cfg.reward_components["knee_strike_prior"].static_params["weight"] = KNEE_STRIKE_WEIGHT
    return cfg
