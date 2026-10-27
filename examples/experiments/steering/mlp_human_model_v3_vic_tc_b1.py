"""vic_TC_b1 (implementation vic_tc_v1, candidate, 2026-10-26): HMV_v1 path B human-gain teacher.
Starts from scratch (--fresh), not from r25 or a VIC checkpoint; delta is released only in a later stage.

Based on r6: hold alpha = 1, beta = 0 and delta = 0 from the first step, so sagittal gains follow the human
reference while Hip_x / Subtalar_x keep their current gains (frontal_human_ref=False). Disable the band and
co-contraction penalties; the metabolic proxy stays at r6's zero weight. Keep the learned feed-forward torque
residual, 79-output actor, effort cost and all other r6 rewards, commands and AMP settings unchanged.
"""
import dataclasses

from examples.experiments.steering.mlp_human_model_v3_vic_tc_r6 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_vic_tc_r6 import (
    env_config as _r6_env_config, agent_config)  # noqa: F401


def env_config(robot_cfg, args):
    cfg = _r6_env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(
        cfg.control_components["steering"],
        alpha_init=1.0, beta_init=0.0, schedule=False,
        delta_max=0.0, delta_band=False, delta_lowpass_s=0.0,
        frontal_human_ref=False, damp_delta_gain=0.5, fresh_stance=True)
    for name in ("vic_band", "vic_cocontraction", "vic_metab"):
        cfg.reward_components[name].static_params["weight"] = 0.0
    return cfg
