"""b2-a7-headroll-s8: b2-a7-headroll with the speed error scale 2 -> 8, nothing else."""
from examples.experiments.steering.mlp_human_model_v3_a7_followups import *


def env_config(robot_cfg, args):
    cfg = add_head_roll(set_stage_b_steering(make_env_config(robot_cfg, args, "a7")), robot_cfg)
    return sharpen_speed_tracking(cfg)
