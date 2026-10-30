"""walk_TC_scratch_ks02_r1 (2026-10-30, slot B; user: "a, b는 seed만 바꾸지 말고 가장 중요해 보이는 weight만 달리해서"):
walk_TC_scratch_r1 (A1a, slot A) with ONE change, knee_strike_prior weight 0.6 -> 0.2, same seed 2944, so slot A vs B
is a clean single-weight ablation of the walking teacher from scratch. Why this weight: it is the largest single reward
weight of the s5 recipe; in the s5-based runs it was saturated (mean 0.96, SD 0.006) yet ~30 % of the reward
(share/training_configs/vic_TC_server/stage_task_reward_inventory_261030.md), the first removal candidate. 0.2, not 0:
from scratch it may still guide the early knee flexion at heel strike. Stage 2 is walk_TC_scratch_ks02_from-walkks02r1_r2.
"""
from examples.experiments.steering.mlp_human_model_v3_walk_tc_scratch_r1 import *  # noqa: F401,F403
from examples.experiments.steering import mlp_human_model_v3_walk_tc_scratch_r1 as _a1a

KNEE_STRIKE_W = 0.2


def env_config(robot_cfg, args):
    cfg = _a1a.env_config(robot_cfg, args)
    cfg.reward_components["knee_strike_prior"].static_params["weight"] = KNEE_STRIKE_W   # as teacher_cadence_v1 r22
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _a1a.agent_config(robot_cfg, env_cfg, args)
