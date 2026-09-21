"""a9: a7 warm start, head world upright and pitch/roll stability."""
from examples.experiments.steering.mlp_human_model_v3_a7_followups import *


def env_config(robot_cfg, args):
    return make_env_config(robot_cfg, args, "a9")
