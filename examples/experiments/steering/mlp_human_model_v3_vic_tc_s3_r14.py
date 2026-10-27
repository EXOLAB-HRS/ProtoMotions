"""vic_TC_s3_r14 (implementation vic_tc_v1 on vic_tc_v1_se, candidate, 2026-10-27): continuation of vic_TC_s3_r13
(old name vic_TC_r13; s3 teacher -> VIC, actor lr 2e-5). r13 ended at alpha = 1, beta = 0 (human stiffness 100 %) with
training falls still dropping (7.5e-4 -> 3.3e-4) and steady speed tracking still rising (0.78 -> 0.83, +4 % of range per
10 % of the run), so the final setting gets another 100M. alpha / beta are read from r13's state file (1.0 / 0.0, the
schedule is already at its end); actor lr 2e-5 is set in the launcher. Everything else is r12 / r12b / r13.
"""
import dataclasses
import json
import os

from examples.experiments.steering.mlp_human_model_v3_vic_tc_r12 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_vic_tc_r12 import (
    env_config as _r12_env_config)

R13_STATE = os.environ.get(
    "HC_VIC_RESUME_STATE",
    "/home/ghlee/human-controller/output/261023/e02_vic_tc_r2_loop/vic-tc-r13.vic_state.json")


def env_config(robot_cfg, args):
    cfg = _r12_env_config(robot_cfg, args)
    s = json.load(open(R13_STATE))
    cfg.control_components["steering"] = dataclasses.replace(
        cfg.control_components["steering"], alpha_init=float(s["alpha"]), beta_init=float(s["beta"]), schedule=True)
    return cfg


def agent_config(robot_cfg, env_cfg, args):  # noqa: F811
    from examples.experiments.steering.mlp_human_model_v3_vic_tc_r12 import agent_config as _r12_agent_config
    cfg = _r12_agent_config(robot_cfg, env_cfg, args)
    # The launcher writes 2e-5 into the forked optimizer state (the value actually used, as in r13 whose recipe still
    # said 5e-5); set it here too so resolved_configs records the real step size.
    cfg.model.actor_optimizer.lr = 2e-5
    return cfg
