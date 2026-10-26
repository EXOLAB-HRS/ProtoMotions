"""teacher_cadence_v1 revision 27 (2026-10-24, baseline-teacher stage 1, ghlee): r25 with one change,
cadence_tracking relative_scale 0.25 -> RELATIVE_SCALE.

r25 at a 1 m/s command walks 1.11 m/s at 1.78 steps/s on all three evaluation seeds (output/261024/e01_tc_baseline_w1w2,
Fukuchi-based W2 check), 10 % above the commanded cadence f* = f0(1.0) = 1.62 and 6 % above the human law at the speed
it actually walks (1.68). With relative_scale 0.25 a 10 % error keeps exp(-(0.10 / 0.25)^2) = 0.85 of the term, so
the cadence term barely pulls; at 0.10 the same error gives exp(-1) = 0.37. Starts from r25 last.ckpt.
"""
from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r25 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r25 import (
    env_config as _parent_env_config, agent_config)  # noqa: F401

RELATIVE_SCALE = 0.10


def env_config(robot_cfg, args):
    cfg = _parent_env_config(robot_cfg, args)
    cfg.reward_components["cadence_tracking"].static_params["relative_scale"] = RELATIVE_SCALE
    return cfg
