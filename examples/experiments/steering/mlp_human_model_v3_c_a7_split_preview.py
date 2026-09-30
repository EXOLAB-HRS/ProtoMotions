"""Stage A-C candidate with one-second braking/turning and shorter acceleration preview."""
from examples.experiments.steering.mlp_human_model_v3_a7_followups import *


def env_config(robot_cfg, args):
    return stage_c_split_preview_recipe(robot_cfg, args)
