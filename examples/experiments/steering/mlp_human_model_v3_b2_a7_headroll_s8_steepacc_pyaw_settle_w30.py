"""b2-a7-headroll-s8-steepacc-pyaw-settle-w30: -settle with the transition term weight 0.15 -> 0.30 (taken from heading)."""
from examples.experiments.steering.mlp_human_model_v3_a7_followups import *


def env_config(robot_cfg, args):
    cfg = add_head_roll(set_stage_b_steep_acceleration(make_env_config(robot_cfg, args, "a7")), robot_cfg)
    cfg = set_transition_until_settled(add_speed_transition(add_pelvis_yaw(sharpen_speed_tracking(cfg))))
    extra = .30 - TRANSITION_WEIGHT
    cfg.reward_components["speed_transition_tracking"].static_params["weight"] = .30
    cfg.reward_components["heading_rew"].static_params["weight"] -= extra
    return cfg
