"""Stage A-C candidate with a one-second bounded command preview."""
from examples.experiments.steering.mlp_human_model_v3_a7_followups import *


def env_config(robot_cfg, args):
    return stage_c_command_preview_recipe(robot_cfg, args, horizon=1.0)
