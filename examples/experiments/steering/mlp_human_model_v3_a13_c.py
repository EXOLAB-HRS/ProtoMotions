"""a13 Stage C: A rehearsal plus acceleration, stop and restart commands."""
from examples.experiments.steering.mlp_human_model_v3_a13_curriculum import *


def env_config(robot_cfg, args):
    return make_env_config(robot_cfg, args, "c")
