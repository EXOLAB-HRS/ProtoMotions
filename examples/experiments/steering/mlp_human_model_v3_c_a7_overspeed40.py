"""A-C overspeed-scale 40 candidate, warm-started from e16 continuation."""
from examples.experiments.steering.mlp_human_model_v3_a7_followups import *


def env_config(robot_cfg, args):
    return stage_c_overspeed_recipe(robot_cfg, args, 40.0)
