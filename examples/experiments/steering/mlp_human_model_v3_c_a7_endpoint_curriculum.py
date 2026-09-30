"""Stage A-C endpoint curriculum candidate warm-started from overspeed40."""
from examples.experiments.steering.mlp_human_model_v3_a7_followups import *


def env_config(robot_cfg, args):
    return stage_c_endpoint_curriculum_recipe(robot_cfg, args)
