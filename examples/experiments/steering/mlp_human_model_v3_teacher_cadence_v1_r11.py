"""teacher_cadence_v1 revision 11: r10 with a stronger, wider human gait-angle prior.

r10 (human_gait_prior 0.10, kernel 1 x SD) fixed the step rate but not the knee at heel strike:
against Lencioni et al. 2019 the knee stays ~25 deg flexed at 0/100 % of the cycle (human ~4 deg).
With SD ~5 deg that error gives exp(-8) ~ 0, so the reward had no slope there.
- human_gait_prior weight 0.10 -> PRIOR_WEIGHT_R11, kernel_scale 1 -> KERNEL_SCALE_R11.
Starts from r10 last.ckpt. Validate human-likeness on Lencioni (training table is Fukuchi).
"""
from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r10 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r10 import (
    env_config as _r10_env_config, agent_config)  # noqa: F401

PRIOR_WEIGHT_R11 = 0.25
KERNEL_SCALE_R11 = 3.0


def env_config(robot_cfg, args):
    cfg = _r10_env_config(robot_cfg, args)
    prior = cfg.reward_components["human_gait_prior"].static_params
    prior["weight"], prior["kernel_scale"] = PRIOR_WEIGHT_R11, KERNEL_SCALE_R11
    return cfg
