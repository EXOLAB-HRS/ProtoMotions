"""Stage A-C deceleration candidate with an overspeed reward tail."""
from examples.experiments.steering.mlp_human_model_v3_a7_followups import *


def env_config(robot_cfg, args):
    return stage_c_overspeed_tail_recipe(robot_cfg, args)
