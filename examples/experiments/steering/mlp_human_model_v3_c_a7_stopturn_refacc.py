"""c-a7-stopturn-refacc: -stopturn-steep with command ramps 0.4-0.8 m/s^2, inside the reference deceleration range."""
from examples.experiments.steering.mlp_human_model_v3_a7_followups import *


def env_config(robot_cfg, args):
    return stage_c_recipe(robot_cfg, args, REFERENCE_ACCELERATION)
