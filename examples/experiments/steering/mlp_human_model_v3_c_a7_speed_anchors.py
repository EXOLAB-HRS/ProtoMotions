"""A-C candidate that samples Stage B speed endpoints more often."""
from examples.experiments.steering.mlp_human_model_v3_a7_followups import *


def env_config(robot_cfg, args):
    return stage_c_speed_anchor_recipe(robot_cfg, args)
