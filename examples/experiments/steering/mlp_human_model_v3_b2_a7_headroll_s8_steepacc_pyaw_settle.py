"""b2-a7-headroll-s8-steepacc-pyaw-settle: -steepacc-pyaw-trans with the transition kept on until speed settles."""
from examples.experiments.steering.mlp_human_model_v3_a7_followups import *


def env_config(robot_cfg, args):
    cfg = add_head_roll(set_stage_b_steep_acceleration(make_env_config(robot_cfg, args, "a7")), robot_cfg)
    return set_transition_until_settled(add_speed_transition(add_pelvis_yaw(sharpen_speed_tracking(cfg))))
