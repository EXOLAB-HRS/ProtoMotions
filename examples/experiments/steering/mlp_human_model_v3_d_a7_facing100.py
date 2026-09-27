"""Method 1 D round 1 candidate: independent facing throughout the turn group."""
from examples.experiments.steering.mlp_human_model_v3_a7_followups import *


def env_config(robot_cfg, args):
    return stage_d_recipe(robot_cfg, args, 1.0)
