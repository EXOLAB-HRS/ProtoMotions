"""teacher_cadence_v1 revision 21: r20 with the AMP reward weight 0.1 -> 0.4 (whole-body naturalness from
the AMASS pack weighs more against the Lencioni leg reward). Starts from r17 last.ckpt, like r20.
"""
from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r20 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r20 import (
    env_config, agent_config as _r20_agent_config)  # noqa: F401

AMP_REWARD_W = 0.4


def agent_config(robot_cfg, env_cfg, args):
    cfg = _r20_agent_config(robot_cfg, env_cfg, args)
    cfg.amp_parameters.discriminator_reward_w = AMP_REWARD_W
    return cfg
