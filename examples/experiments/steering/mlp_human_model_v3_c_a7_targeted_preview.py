"""Stage A-C candidate with distinct stop, walking-brake and turn preview."""
from examples.experiments.steering.mlp_human_model_v3_c_a7_split_preview import *
from examples.experiments.steering.mlp_human_model_v3_a7_followups import stage_c_targeted_preview_recipe


def env_config(robot_cfg, args):
    return stage_c_targeted_preview_recipe(robot_cfg, args)
