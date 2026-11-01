"""vic4_TC_s5_from-s5_noeff-sub_r1 (server, 2026-11-01): vic4_TC_s5_from-s5_noeff_r1 (slot B, 4096 env) with the beta prior per
physics substep (VicSteeringConfig.v4_prior_substep, dynamics._vic_prior_substep). Reason: the held control-step prior is not
equivalent to the fixed PD (fork-initialised s5 at alpha 1 / beta 1: 93 % fall ticks held vs 0-1.4 % substep, knee r
0.95-0.97, local check 2026-11-01), and server noeff kept the gait at 0.8 / 1.0 m/s (knee r 0.98) but not at 1.4 m/s (0.73).
Standard ramp (9000 + 9000 control steps); at 4096 env this gives the same data per ramp stage as the local
vic4_TC_s5_from-s5_noeff-ramp2x-sub_r1 (2048 env, ramp 18000). Only change against noeff.
"""
import dataclasses

from examples.experiments.steering import mlp_human_model_v3_vic4_tc_s5_from_s5_noeff_r1 as _noeff
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_s5_from_s5_noeff_r1 import *  # noqa: F401,F403


def env_config(robot_cfg, args):
    cfg = _noeff.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(cfg.control_components["steering"], v4_prior_substep=True)
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _noeff.agent_config(robot_cfg, env_cfg, args)
