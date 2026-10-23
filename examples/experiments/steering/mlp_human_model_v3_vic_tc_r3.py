"""vic_TC_r3 (implementation vic_tc_v1, candidate change, 2026-10-23): vic_TC_r2 plus gait-kinematics weight, run in
parallel with r2 on a second GPU. Starts from r1 last.ckpt like r2.

W2 check of r1 (output/261023/e02_vic_tc_r2_loop/configs/check_criteria.py, 1 m/s): cadence 1.98 steps/s vs r25 1.78
and the human law 1.71; right ankle curve r 0.54 vs Fukuchi (r25 0.88), knee r 0.89 (r25 0.97). The soft joints
moved the gait away from the reference that r25 had learned, so r3 raises the two terms that set it:
- human_gait_prior (Lencioni hip/knee/ankle curves) weight x PRIOR_GAIN;
- cadence_tracking (step period vs the commanded cadence, logged at weight 0 since teacher_cadence_v1) to
  CADENCE_WEIGHT.
"""
from examples.experiments.steering.mlp_human_model_v3_vic_tc_r2 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_vic_tc_r2 import (
    env_config as _r2_env_config, agent_config)  # noqa: F401

PRIOR_GAIN = 1.6
CADENCE_WEIGHT = 0.10


def env_config(robot_cfg, args):
    cfg = _r2_env_config(robot_cfg, args)
    cfg.reward_components["human_gait_prior"].static_params["weight"] *= PRIOR_GAIN
    cfg.reward_components["cadence_tracking"].static_params["weight"] = CADENCE_WEIGHT
    return cfg
