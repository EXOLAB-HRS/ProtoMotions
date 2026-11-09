"""Walk teacher on human_model_v3.6: vic4_tc_walk33_v36_strat_spd_up_sw20_ha0_wang_ua01_r1 aimed at the kinetics, head and arms too.

404 (sw20_ha0_wang_ua01, seed 2948) passes W2 v5, W3 and MoS but walks with a hip strategy (1.0 m/s push-off 0.94 vs human 1.47,
late-stance hip minimum -0.54 vs -0.27 N m/kg), nods the head +3 SD (pitch range 13 deg) and swings the arms +5 SD. User
2026-11-09: W1, W2, W3, W2-2k and head / arm must be trained together, not one after the other. Weights only:
  vic_strategy_cost 0.3 -> 0.6 (300 e483 at 0.3 from a slower parent reached push-off 1.38 / hip -0.33; 0.9 in 350 collapsed speed),
  head_angular_stability 0.0 -> 0.08 (about half of the 0.15 that cost W2 / MoS before ha0),
  vic_upper_prior 0.1 -> 0.2 (0.3 slowed every speed in 386; steady speed weight 2.0 now holds speed).
"""
from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_up_sw20_ha0_wang_ua01_r1 as _base
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_up_sw20_ha0_wang_ua01_r1 import *  # noqa: F401,F403

STRATEGY_WEIGHT = 0.6
HEAD_RATE_WEIGHT = 0.08
UPPER_WEIGHT = 0.2


def env_config(robot_cfg, args):
    cfg = _base.env_config(robot_cfg, args)
    cfg.reward_components["vic_strategy_cost"].static_params["weight"] = STRATEGY_WEIGHT
    cfg.reward_components["head_angular_stability"].static_params["weight"] = HEAD_RATE_WEIGHT
    cfg.reward_components["vic_upper_prior"].static_params["weight"] = UPPER_WEIGHT
    return cfg
