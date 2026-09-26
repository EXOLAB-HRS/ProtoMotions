"""c-a7-stopturn-steep: method 1 Stage C, the 260925 e04 recipe plus stops and turns, command ramps 1.6-3.2 m/s^2."""
from examples.experiments.steering.mlp_human_model_v3_a7_followups import *


def env_config(robot_cfg, args):
    return stage_c_recipe(robot_cfg, args, STEEP_ACCELERATION)
