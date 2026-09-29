"""b2-a7-headroll-s8-fastacc: b2-a7-headroll-s8 with command ramps 0.4-0.8 -> 0.8-1.6 m/s^2, nothing else."""
from examples.experiments.steering.mlp_human_model_v3_a7_followups import *


def env_config(robot_cfg, args):
    cfg = add_head_roll(set_stage_b_fast_acceleration(make_env_config(robot_cfg, args, "a7")), robot_cfg)
    return sharpen_speed_tracking(cfg)
