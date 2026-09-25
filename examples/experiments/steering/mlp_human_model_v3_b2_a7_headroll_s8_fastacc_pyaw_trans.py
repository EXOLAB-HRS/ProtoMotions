"""b2-a7-headroll-s8-fastacc-pyaw-trans: -fastacc-pyaw plus speed tracking scored during speed transitions."""
from examples.experiments.steering.mlp_human_model_v3_a7_followups import *


def env_config(robot_cfg, args):
    cfg = add_head_roll(set_stage_b_fast_acceleration(make_env_config(robot_cfg, args, "a7")), robot_cfg)
    return add_speed_transition(add_pelvis_yaw(sharpen_speed_tracking(cfg)))
