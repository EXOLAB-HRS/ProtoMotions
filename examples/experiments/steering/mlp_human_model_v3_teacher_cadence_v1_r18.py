"""teacher_cadence_v1 revision 18: r17 with the discriminator gradient penalty 5 -> 20.

r16/r17 (AMP weight 0.4): the discriminator still separates agent from Lencioni pack frames at 0.98
accuracy (AMP reward ~0.10), so the style reward is nearly flat. A larger gradient penalty smooths the
discriminator and gives the policy a usable style gradient. Starts from r17 last.ckpt.
"""
from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r17 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r17 import (
    env_config, agent_config as _r17_agent_config)  # noqa: F401

DISCRIMINATOR_GRAD_PENALTY = 20.0


def agent_config(robot_cfg, env_cfg, args):
    cfg = _r17_agent_config(robot_cfg, env_cfg, args)
    cfg.amp_parameters.discriminator_grad_penalty = DISCRIMINATOR_GRAD_PENALTY
    return cfg
