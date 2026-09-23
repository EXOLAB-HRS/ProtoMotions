"""a7-headroll: Stage A commands plus the head roll term, from a7.

Isolates the head term on the task a7 already passes, so its effect on head
motion and on Stage A tracking is read without the Stage B command change.
"""
from examples.experiments.steering.mlp_human_model_v3_a7_followups import *


def env_config(robot_cfg, args):
    return add_head_roll(make_env_config(robot_cfg, args, "a7"), robot_cfg)
