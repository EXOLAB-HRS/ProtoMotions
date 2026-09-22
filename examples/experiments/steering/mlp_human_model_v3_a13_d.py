"""a13 Stage D: replay plus full travel headings and independent facing."""
from examples.experiments.steering.mlp_human_model_v3_a13_curriculum import *


def env_config(robot_cfg, args):
    return make_env_config(robot_cfg, args, "d")
