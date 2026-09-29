"""Stage A-C candidate emphasizing the 1.4 to 1.0 m/s braking transition."""
from examples.experiments.steering.mlp_human_model_v3_a7_followups import *


def env_config(robot_cfg, args):
    return stage_c_pair_curriculum_recipe(robot_cfg, args)
