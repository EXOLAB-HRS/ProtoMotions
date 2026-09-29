"""Stage D small relative-facing candidate; no implementation version change."""
from examples.experiments.steering.mlp_human_model_v3_a7_followups import *


def env_config(robot_cfg, args):
    return stage_d_relative_facing_recipe(robot_cfg, args)
