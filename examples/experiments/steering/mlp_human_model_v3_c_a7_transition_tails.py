"""Stage A-C candidate retaining speed-error gradients on braking and restart."""
from examples.experiments.steering.mlp_human_model_v3_a7_followups import *


def env_config(robot_cfg, args):
    return stage_c_transition_tails_recipe(robot_cfg, args)
