"""vic_TC_r4 (implementation vic_tc_v1, candidate change, 2026-10-23): vic_TC_r2 with the damping ratio held while the
policy changes stiffness (k_d = exp(lb + 0.5 delta), so k_d ~ sqrt(k_p) and zeta stays at the human reference),
instead of r1/r2's k_d ~ k_p (zeta grows as sqrt(k_p)). Only this differs from r2, so r4 vs r2 isolates the damping
law (user question 2026-10-23 on critical damping; full critical damping zeta = 1 would put knee swing damping 10-70x
above human measurements). Starts from r1 last.ckpt like r2.
"""
import dataclasses

from examples.experiments.steering.mlp_human_model_v3_vic_tc_r2 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_vic_tc_r2 import (
    env_config as _r2_env_config, agent_config)  # noqa: F401

DAMP_DELTA_GAIN = 0.5


def env_config(robot_cfg, args):
    cfg = _r2_env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(
        cfg.control_components["steering"], damp_delta_gain=DAMP_DELTA_GAIN)
    return cfg
