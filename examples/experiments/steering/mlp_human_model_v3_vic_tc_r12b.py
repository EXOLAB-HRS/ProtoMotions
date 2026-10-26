"""vic_TC_r12b (implementation vic_tc_v1 on vic_tc_v1_se, candidate, 2026-10-27): continuation of vic_TC_r12 (se-v1-s3 ->
VIC). r12 (100M) spent its run in the alpha curriculum: at 30 % the fall gate held alpha near 0.5-0.57 (falls 7e-4 vs
s3 1e-4), so the final setting (alpha = 1, beta = 0) gets little or no training, as in r1 (only its last 24 %).
r12b starts from r12 last.ckpt and resumes the schedule where r12 stopped: alpha_init / beta_init are read from r12's
state file (HC_VIC_STATE_FILE of the r12 run, written every log step), the fall gate is re-baselined on the first 600
steps (at the resumed alpha), and alpha -> 1 then beta -> 0 continue on the r1 step schedule. Everything else is r12.
"""
import json
import os

from examples.experiments.steering.mlp_human_model_v3_vic_tc_r12 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_vic_tc_r12 import (
    env_config as _r12_env_config, agent_config)  # noqa: F401
import dataclasses

R12_STATE = os.environ.get(
    "HC_VIC_RESUME_STATE",
    "/home/ghlee/human-controller/output/261023/e02_vic_tc_r2_loop/vic-tc-r12.vic_state.json")


def _resume():
    try:
        s = json.load(open(R12_STATE))
        return float(s["alpha"]), float(s["beta"])
    except (OSError, KeyError, ValueError):
        return 1.0, 1.0


def env_config(robot_cfg, args):
    cfg = _r12_env_config(robot_cfg, args)
    a, b = _resume()
    cfg.control_components["steering"] = dataclasses.replace(
        cfg.control_components["steering"], alpha_init=a, beta_init=b, schedule=True)
    return cfg
