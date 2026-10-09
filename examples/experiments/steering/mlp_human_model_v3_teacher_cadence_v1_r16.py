"""teacher_cadence_v1 revision 16: r15 (Lencioni 2019 AMP pack) with the AMP reward weight 0.1 -> 0.4.

r15 kept r12's gait (knee at contact 26.6 deg, same definition as r12's 26.4): since r7 the AMP reward
weight is 0.1 against task_reward_w 0.6, and the discriminator separates agent from pack frames at 0.98
accuracy, so the style signal is ~0.09. 0.4 is the value before r7. Starts from r15 last.ckpt (its
discriminator is already trained on the Lencioni pack).
"""
from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r15 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r15 import (
    env_config, agent_config as _r15_agent_config)  # noqa: F401

AMP_REWARD_W = 0.4


def agent_config(robot_cfg, env_cfg, args):
    cfg = _r15_agent_config(robot_cfg, env_cfg, args)
    cfg.amp_parameters.discriminator_reward_w = AMP_REWARD_W
    return cfg
