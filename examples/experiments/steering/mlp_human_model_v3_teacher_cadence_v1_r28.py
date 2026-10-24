"""teacher_cadence_v1 revision 28 (2026-10-24, baseline-teacher stage 1, ghlee): r25 with one change,
human_gait_prior ratio_tol 0.05 -> RATIO_TOL.

r25 vs Fukuchi 2018 at 1.1 m/s (right leg, three evaluation seeds identical): swing knee peak 48 vs 62 deg
(-17 deg at 70-79 %, z -4.1), ankle at heel strike -7 vs +1 deg, loading response -13.5 deg below human (z -4.4), no
swing dorsiflexion (max -2.4 vs +6 deg). The Lencioni prior that scores these curves only acts when the commanded
cadence ratio is within 1 +/- 0.05 (r12), i.e. in cadence_natural_fraction = 0.3 of the commands. RATIO_TOL covers the
whole commanded range 0.8-1.25 so the prior acts on every walking step. Starts from r25 last.ckpt.
"""
from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r25 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r25 import (
    env_config as _parent_env_config, agent_config)  # noqa: F401

RATIO_TOL = 0.30


def env_config(robot_cfg, args):
    cfg = _parent_env_config(robot_cfg, args)
    cfg.reward_components["human_gait_prior"].static_params["ratio_tol"] = RATIO_TOL
    return cfg
