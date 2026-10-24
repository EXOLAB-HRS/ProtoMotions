"""vic_TC_r6 (implementation vic_tc_v1, candidate change, 2026-10-24): vic_TC_r4 without the metabolic proxy cost.
Starts from r1 last.ckpt like r2-r5.

r4 (damping ratio held) passes W3 and W4 (energy ratio 2.6-2.8 / 2.2-2.3, mech 0.63-0.65, ankle work share 0.44) but
W1/W2 stay below r1 (falls 0.38/min, knee r 0.71, ankle r 0.04), as in r2/r3, which all carry vic_metab 0.03. r6 drops
that term (weight 0) and keeps r4's damping law and r2's heading x1.2, so r6 vs r4 isolates the metabolic cost.
"""
from examples.experiments.steering.mlp_human_model_v3_vic_tc_r4 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_vic_tc_r4 import (
    env_config as _r4_env_config, agent_config)  # noqa: F401

METAB_WEIGHT = 0.0


def env_config(robot_cfg, args):
    cfg = _r4_env_config(robot_cfg, args)
    cfg.reward_components["vic_metab"].static_params["weight"] = METAB_WEIGHT
    return cfg
