"""b2-a7: a13 stage-b commands on the a7 recipe, no head term.

The Stage B control for b2-a7-headroll. Unlike b1-a7 it keeps to the speed range
the reference supports (0.7-1.4 m/s) and leaves stopping and turning to C and D.
"""
from examples.experiments.steering.mlp_human_model_v3_a7_followups import *


def env_config(robot_cfg, args):
    return set_stage_b_steering(make_env_config(robot_cfg, args, "a7"))
