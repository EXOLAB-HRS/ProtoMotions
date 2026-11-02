"""vic4_TC_s5_from-noeffsube460_slowramp_r1 (server, 2026-11-02; comparison arm of the beta stair): parent
vic4_TC_s5_from-s5_noeff-sub_r1 epoch_460 (c0076f63, train beta 0.333), alpha 1, beta 0.33 -> 0 continuously over the first
45M of a 60M run, then beta 0 for the last 15M. beta_steps = round(45e6 / 4096 env / 0.33) = 33292 control steps per unit beta
(the previous run moved beta 1 -> 0 in 9000). The ramp starts after the 600-step fall baseline and pauses while the training
fall rate is above tolerance (vic_steering hold). Otherwise the same as vic4_TC_s5_from-s5_noeff-sub_r1.
"""
import dataclasses

from examples.experiments.steering import mlp_human_model_v3_vic4_tc_s5_from_s5_noeff_sub_r1 as _sub
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_s5_from_s5_noeff_sub_r1 import *  # noqa: F401,F403

BETA_STEPS = 33292


def env_config(robot_cfg, args):
    cfg = _sub.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(cfg.control_components["steering"], alpha_init=1.0,
                                                             beta_init=0.33, schedule=True, beta_steps=BETA_STEPS)
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _sub.agent_config(robot_cfg, env_cfg, args)
