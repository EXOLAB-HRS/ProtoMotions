"""c-a7-stops-refacc: Stage C with stops and restarts only (no turns), command ramps 0.4-0.8 m/s^2.

The AMP pack is chosen at launch (--motion): a7's forward pack or the Stage C pack.
"""
from examples.experiments.steering.mlp_human_model_v3_a7_followups import *


def env_config(robot_cfg, args):
    return stage_c_stops_recipe(robot_cfg, args)
