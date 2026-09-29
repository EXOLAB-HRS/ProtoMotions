"""b2-a7-headroll-s8-steepacc-pyaw: the -fastacc-pyaw candidate with command ramps 0.8-1.6 -> 1.6-3.2 m/s^2."""
from examples.experiments.steering.mlp_human_model_v3_a7_followups import *


def env_config(robot_cfg, args):
    cfg = add_head_roll(set_stage_b_steep_acceleration(make_env_config(robot_cfg, args, "a7")), robot_cfg)
    return add_pelvis_yaw(sharpen_speed_tracking(cfg))
