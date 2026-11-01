"""vic4_TC_s5_from-noeffr1_r2 (server, 2026-11-01; user "탐색보다 빨리 좋은 결과"): continuation of the server run
vic4_TC_s5_from-s5_noeff_r1 (slot B, seed 2944, 80M last cc6a2b7f, stopped by the server shutdown right after its ramp;
knee r 0.981 / 0.979 / 0.728 at 0.8 / 1.0 / 1.4 m/s, re-evaluated locally) at the final law with no ramp (alpha 1, beta 0,
schedule off) so the reloaded ramp does not restart. Same rewards and settings: a length extension of the same run.
"""
import dataclasses

from examples.experiments.steering import mlp_human_model_v3_vic4_tc_s5_from_s5_noeff_r1 as _noeff
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_s5_from_s5_noeff_r1 import *  # noqa: F401,F403


def env_config(robot_cfg, args):
    cfg = _noeff.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(cfg.control_components["steering"],
                                                             alpha_init=1.0, beta_init=0.0, schedule=False)
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _noeff.agent_config(robot_cfg, env_cfg, args)
