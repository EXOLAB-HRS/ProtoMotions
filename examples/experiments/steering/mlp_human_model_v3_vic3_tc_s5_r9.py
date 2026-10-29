"""vic3_TC_s5_r9 (vic_tc_v3 candidate, 2026-10-29, local PC) = vic3_TC_s5_r4 with pulses twice as large
(v3_perturb_amp 0.08 -> 0.16 of the effort limit). Question: does u follow the size of the unpredictable load
(people co-contract more under larger instability, Burdet et al. 2001), or does nothing move u? Same start as r4.
"""
import dataclasses

from examples.experiments.steering import mlp_human_model_v3_vic3_tc_s5_r4 as _v3r4
from examples.experiments.steering.mlp_human_model_v3_vic3_tc_s5_r4 import *  # noqa: F401,F403


def env_config(robot_cfg, args):
    cfg = _v3r4.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(cfg.control_components["steering"], v3_perturb_amp=0.16)
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _v3r4.agent_config(robot_cfg, env_cfg, args)
