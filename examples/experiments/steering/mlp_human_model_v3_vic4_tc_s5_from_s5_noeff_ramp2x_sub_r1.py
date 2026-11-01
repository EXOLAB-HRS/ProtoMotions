"""vic4_TC_s5_from-s5_noeff-ramp2x-sub_r1 (local, 2026-11-01): vic4_TC_s5_from-s5_noeff-ramp2x_r1 with the beta prior evaluated
per physics substep (VicSteeringConfig.v4_prior_substep, dynamics._vic_prior_substep). Reason: the held control-step prior
is not equivalent to the fixed PD (fork-initialised s5 at alpha 1 / beta 1 fell on 93 % of ticks), while the substep prior
is (same check: 0-1.4 % falls, knee r 0.95-0.97, ankle r 0.85-0.92 at 0.8 / 1.0 / 1.4 m/s), so the alpha / beta ramp
starts from the walker's own torque and each stage is a small change. Only change against ramp2x (same seed 2944).
"""
import dataclasses

from examples.experiments.steering import mlp_human_model_v3_vic4_tc_s5_from_s5_noeff_ramp2x_r1 as _r2x
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_s5_from_s5_noeff_ramp2x_r1 import *  # noqa: F401,F403


def env_config(robot_cfg, args):
    cfg = _r2x.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(cfg.control_components["steering"], v4_prior_substep=True)
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _r2x.agent_config(robot_cfg, env_cfg, args)
