"""a7-headroll-s8: a7-headroll with the speed error scale 2 -> 8, nothing else."""
from examples.experiments.steering.mlp_human_model_v3_a7_followups import *


def env_config(robot_cfg, args):
    return sharpen_speed_tracking(add_head_roll(make_env_config(robot_cfg, args, "a7"), robot_cfg))
