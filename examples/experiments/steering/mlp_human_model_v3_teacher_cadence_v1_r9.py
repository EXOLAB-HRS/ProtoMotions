"""teacher_cadence_v1 revision 9: r7 recipe with a human f0(v) and a restored speed-tracking weight.

Starts from r8 (r7 recipe, 150M), which follows rho* but walks faster than commanded.
- f0(v) = 0.608 * v + 1.178 steps/s, the linear fit of Fukuchi et al. 2018 treadmill trials
  (42 subjects, 309 trials; scripts/human_gait_ref_compare.py, 2026-10-06). The old f0 was the parent
  policy's own cadence, 6-15% above the human fit at 0.8-1.2 m/s.
- heading_rew weight 0.15 -> 0.30. The parent pc_240_4_r3 had 0.45; the cadence terms took 0.30 of it,
  and r8 overshoots speed (Stage A forward MAE 0.06 -> 0.12 m/s).
Discriminator labels are unchanged: they are normalised to the pack's own median, not to f0.
"""
from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r7 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r7 import (
    env_config as _r7_env_config, agent_config)  # noqa: F401

F0_SLOPE_HUMAN, F0_INTERCEPT_HUMAN = 0.608, 1.178
HEADING_WEIGHT_R9 = 0.30


def env_config(robot_cfg, args):
    cfg = _r7_env_config(robot_cfg, args)
    steering = cfg.control_components["steering"]
    steering.cadence_f0_slope, steering.cadence_f0_intercept = F0_SLOPE_HUMAN, F0_INTERCEPT_HUMAN
    cfg.reward_components["heading_rew"].static_params["weight"] = HEADING_WEIGHT_R9
    return cfg
