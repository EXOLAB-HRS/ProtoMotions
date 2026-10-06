"""teacher_cadence_v1 revision 7: r6 with the AMP reward weight 0.4 -> 0.1 (discriminator ablation).

r3-r6 learned no rho* response although the clock and cadence measurement are correct (2026-10-04
trace check) and r6 gave the tracking reward a usable slope. Remaining hypothesis: the
cadence-conditioned discriminator penalises a 20% cadence change more than the cadence rewards
(0.6 x 0.30) pay for it. Only discriminator_reward_w changes; task_reward_w stays 0.6.
"""
from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r6 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r6 import (
    env_config, agent_config as _r6_agent_config)  # noqa: F401

AMP_REWARD_W = 0.1


def agent_config(robot_cfg, env_cfg, args):
    cfg = _r6_agent_config(robot_cfg, env_cfg, args)
    cfg.amp_parameters.discriminator_reward_w = AMP_REWARD_W
    return cfg
