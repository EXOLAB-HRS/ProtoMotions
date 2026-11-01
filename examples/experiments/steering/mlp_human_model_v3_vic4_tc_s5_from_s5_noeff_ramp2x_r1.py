"""vic4_TC_s5_from-s5_noeff-ramp2x_r1 (local, 2026-11-01): vic4_TC_s5_from-s5_noeff_r1 with the alpha 0 -> 1 / beta 1 -> 0 ramp
twice as long in control steps (9000 -> 18000 each). Reason: the same noeff recipe kept the gait on the server (4096 env,
knee r 0.98 at 0.8 / 1.0 m/s, re-evaluated locally) but collapsed locally (2048 env, e687 just after beta 0: knee r 0.86 /
0.64 / 0.02); the only config differences are env / batch and seed (resolved_configs diff). The ramp is counted in control
steps, so with half the envs each ramp stage gives the policy half the data; doubling the ramp equalises the data per stage.
Run with seed 2944 (= the server run) to reproduce it locally.
"""
import dataclasses

from examples.experiments.steering import mlp_human_model_v3_vic4_tc_s5_from_s5_noeff_r1 as _noeff
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_s5_from_s5_noeff_r1 import *  # noqa: F401,F403

RAMP_STEPS = 18000


def env_config(robot_cfg, args):
    cfg = _noeff.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(cfg.control_components["steering"],
                                                             alpha_steps=RAMP_STEPS, beta_steps=RAMP_STEPS)
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _noeff.agent_config(robot_cfg, env_cfg, args)
