"""b2-a7-headroll: method 1 Stage B candidate — stage-b commands plus the head roll term."""
from examples.experiments.steering.mlp_human_model_v3_a7_followups import *


def env_config(robot_cfg, args):
    return add_head_roll(set_stage_b_steering(make_env_config(robot_cfg, args, "a7")), robot_cfg)
