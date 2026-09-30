"""A-C candidate for the 0.5 rad/s C3 turning condition."""
from examples.experiments.steering.mlp_human_model_v3_a7_followups import *


def env_config(robot_cfg, args):
    return stage_c_slow_turn_recipe(robot_cfg, args)
