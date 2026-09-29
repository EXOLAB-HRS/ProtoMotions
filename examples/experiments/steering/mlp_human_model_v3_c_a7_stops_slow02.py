"""c-a7-stops-slow02: c-a7-stops-refacc with moving commands sampled from 0.2-1.4 m/s instead of 0.7-1.4."""
from examples.experiments.steering.mlp_human_model_v3_a7_followups import *


def env_config(robot_cfg, args):
    return stage_c_stops_slow_recipe(robot_cfg, args, 0.2)
