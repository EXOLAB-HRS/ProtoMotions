"""vic4_TC_s5_r3 (vic_tc_v4, 2026-10-30, local) = vic4_TC_s5_r1 + the afferent delay of vic2_TC_s5_r13 (proprioception
U(40, 70) ms, vestibular U(30, 60) ms on the actor only; critic and discriminators undelayed). Control for the user's
question whether the teacher can stay undelayed: same law, parent and length as the server r1 / r2.
"""
from protomotions.envs.obs.sensory_delay import SensoryDelayConfig, delay_actor_inputs

from examples.experiments.steering import mlp_human_model_v3_vic4_tc_s5_r1 as _v4r1
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_s5_r1 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_vic2_tc_s5_r13 import SENSORY_DELAY


def env_config(robot_cfg, args):
    cfg = _v4r1.env_config(robot_cfg, args)
    cfg.sensory_delay = SensoryDelayConfig(**SENSORY_DELAY)
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    cfg = _v4r1.agent_config(robot_cfg, env_cfg, args)
    return delay_actor_inputs(cfg, env_cfg.sensory_delay)
