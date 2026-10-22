"""vic_TC_r1_slow (candidate, 2026-10-22): vic_tc_r1 with alpha and beta each moved over 15000 control steps
(r1: 9000) and a 150M budget. Hedge for r1 stalling on the fall-rate gate before beta reaches 0.
"""
import dataclasses

from examples.experiments.steering import mlp_human_model_v3_vic_tc_r1 as _vic
from examples.experiments.steering.mlp_human_model_v3_vic_tc_r1 import *  # noqa: F401,F403

SCHEDULE_STEPS = 15000
agent_config = _vic.agent_config


def env_config(robot_cfg, args):
    cfg = _vic.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(
        cfg.control_components["steering"], alpha_steps=SCHEDULE_STEPS, beta_steps=SCHEDULE_STEPS)
    return cfg
