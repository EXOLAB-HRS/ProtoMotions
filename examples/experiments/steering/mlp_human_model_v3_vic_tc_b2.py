"""vic_TC_b2 (implementation vic_tc_v1, candidate, 2026-10-26): HMV_v1 path B human-gain teacher, vic_TC_b1 with
frontal_human_ref=True (user, after share/HMV_v1_teacher_261026/stage_vic_stiffness_261026.md: A, B and C use the
same stiffness reference). Hip_x / Subtalar_x follow HUMAN_RANGES_FRONTAL instead of the current gain, so with
delta = 0 every VIC joint holds a human-scale gain from the first step. b1 was stopped during its smoke.
"""
import dataclasses

from examples.experiments.steering.mlp_human_model_v3_vic_tc_b1 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_vic_tc_b1 import (
    env_config as _b1_env_config, agent_config)  # noqa: F401


def env_config(robot_cfg, args):
    cfg = _b1_env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(cfg.control_components["steering"],
                                                             frontal_human_ref=True)
    return cfg
