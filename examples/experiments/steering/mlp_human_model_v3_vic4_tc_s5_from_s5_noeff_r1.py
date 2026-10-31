"""vic4_TC_s5_from-s5_noeff_r1 (scratch strategy A stage A2 candidate, 2026-11-01; user: server ends 08 KST, results fast,
"s5에 A2"): the A2 recipe of vic4_TC_walkA1_from-walkr2_r1 (walker reward set = teacher_se_v1 s5's, knee_strike 0.6 = s5's)
started from the frozen fixed-PD teacher se-v1-s5 last, with the VIC effort cost removed. Reason: the A1 walkers learned
their gait without any torque cost; local A2 s2946 (effort 0.05 added at A2) kept falls low but rebuilt the gait (knee r
0.97 -> 0.56, ankle 0.84 -> 0.41, knee 23 deg at 1.4 m/s strike), so the new cost is the first suspect. Everything else unchanged.
"""
from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walka1_from_walkr2_r1 as _a2
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walka1_from_walkr2_r1 import *  # noqa: F401,F403


def env_config(robot_cfg, args):
    cfg = _a2.env_config(robot_cfg, args)
    cfg.reward_components.pop("vic_effort")
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _a2.agent_config(robot_cfg, env_cfg, args)
