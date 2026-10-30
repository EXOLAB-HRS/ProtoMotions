"""vic4_TC_scratch_from-vic4cr2_r3 (scratch strategy A, stage A2, candidate 2026-10-30): the A1b recipe with the stiffness
outputs unfrozen (v4_freeze_a False): the policy moves k inside twice the human range, the dead-zone impedance prior
(0.1) keeps the total stiffness human-like. The launcher zeroes the 10 stiffness rows of the A1b actor at the fork and
sets their log std to -1.2, so the first actions are a = 0 (A1b's gait and impedance). Band cost stays removed.
"""
import dataclasses

from examples.experiments.steering import mlp_human_model_v3_vic4c_tc_scratch_from_vic4cr1_r2 as _a1b
from examples.experiments.steering.mlp_human_model_v3_vic4c_tc_scratch_from_vic4cr1_r2 import *  # noqa: F401,F403


def env_config(robot_cfg, args):
    cfg = _a1b.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(cfg.control_components["steering"], v4_freeze_a=False)
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _a1b.agent_config(robot_cfg, env_cfg, args)
