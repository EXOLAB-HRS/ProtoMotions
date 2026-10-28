"""vic_TC_b6 (implementation vic_tc_v1 on vic_tc_v1_se, candidate, 2026-10-28): HMV_v1 path B, last run of the b3-b6 cap.
vic_TC_b4 last.ckpt (cb82cccd..., 80M) continued with vic_TC_b5's pelvis gait prior (0.1) and cadence_tracking 0.2 -> 0.3.

Why: b5 (b4 + pelvis prior) passed pelvis obliquity at all speeds at 40M, but walked a fast shuffle at 1.4 m/s
(cadence 2.53 vs human 2.03 steps/s, ankle curve r 0.4, swing knee asymmetry 25) and at 60M lost quiet standing (10/64).
b4 80M passed cadence and quiet standing, so b6 starts again from b4 last. Codex review 2026-10-28: raise only the
existing cadence term; se-v1-c04 (0.2 -> 0.4) changed little, so the effect is a hypothesis. Everything else as b5.
"""
from examples.experiments.steering.mlp_human_model_v3_vic_tc_b5 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_vic_tc_b5 import env_config as _b5_env_config

CADENCE_WEIGHT_B6 = 0.3


def env_config(robot_cfg, args):
    cfg = _b5_env_config(robot_cfg, args)
    cfg.reward_components["cadence_tracking"].static_params["weight"] = CADENCE_WEIGHT_B6
    return cfg
