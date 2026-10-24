"""vic_TC_r8 (implementation vic_tc_v1, candidate change, 2026-10-24): vic_TC_r6 with r1's heading weight.
Starts from r1 last.ckpt like r2-r7.

r6 (r4 without the metabolic cost) still passes W3 and W4 (energy ratio 3.2-3.4 / 2.7-2.8) but falls 0.42/min vs r1
0.24, so the metabolic cost is not what made r2-r6 fall more. Besides it, r2-r6 share r2's heading_rew x1.2
(HEADING_GAIN) and the fixed alpha = 1, beta = 0 start with fresh_stance. r8 undoes the heading gain only, so r8 vs r6
isolates it.
"""
from examples.experiments.steering.mlp_human_model_v3_vic_tc_r6 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_vic_tc_r6 import (
    env_config as _r6_env_config, agent_config, HEADING_GAIN)  # noqa: F401


def env_config(robot_cfg, args):
    cfg = _r6_env_config(robot_cfg, args)
    cfg.reward_components["heading_rew"].static_params["weight"] /= HEADING_GAIN
    return cfg
