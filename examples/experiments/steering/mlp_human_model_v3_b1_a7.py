"""b1-a7: a7 warm start, rate-limited speed and heading commands."""
from examples.experiments.steering.mlp_human_model_v3_a7_followups import *


def env_config(robot_cfg, args):
    return make_env_config(robot_cfg, args, "b1-a7")
