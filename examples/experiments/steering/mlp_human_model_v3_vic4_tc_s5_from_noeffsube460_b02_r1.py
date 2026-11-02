"""vic4_TC_s5_from-noeffsube460_b02_r1 (server, 2026-11-02; user-approved plan 1 "beta 계단식 하강"): stair A-1: parent vic4_TC_s5_from-s5_noeff-sub_r1 epoch_460 (c0076f63, train beta 0.333), beta fixed 0.2, 30M.
Same as vic4_TC_s5_from-s5_noeff-sub_r1 (noeff, v4_prior_substep True); only alpha_init 1, beta_init 0.2, schedule off
(beta held). The training-fall hold of vic_steering cannot see the gait collapse (held_steps 0 in the last run), so each beta
level is its own run with an HMV-Walk gate in between (knee r min >= 0.90, W2 v5, initiation falls <= 0.05).
"""
import dataclasses

from examples.experiments.steering import mlp_human_model_v3_vic4_tc_s5_from_s5_noeff_sub_r1 as _sub
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_s5_from_s5_noeff_sub_r1 import *  # noqa: F401,F403

BETA = 0.2


def env_config(robot_cfg, args):
    cfg = _sub.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(cfg.control_components["steering"],
                                                             alpha_init=1.0, beta_init=BETA, schedule=False)
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _sub.agent_config(robot_cfg, env_cfg, args)
