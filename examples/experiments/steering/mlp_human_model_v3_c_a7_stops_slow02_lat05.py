"""c-a7-stops-slow02-lat05: c-a7-stops-slow02 with the heading reward's perpendicular-velocity weight 0.5 (was 0.1)."""
from examples.experiments.steering.mlp_human_model_v3_a7_followups import *


def env_config(robot_cfg, args):
    return stage_c_stops_slow_lateral_recipe(robot_cfg, args, 0.5)
