"""a10: a7 warm start, processed-target second-difference regularization."""
from examples.experiments.steering.mlp_human_model_v3_a7_followups import *


def env_config(robot_cfg, args):
    return make_env_config(robot_cfg, args, "a10")
