"""teacher_cadence_v1 revision 12: r11 with the human gait prior gated to natural cadence and a
heavier knee term at heel strike.

r11 (prior 0.25, kernel 3 x SD) against Lencioni et al. 2019: knee still ~23 deg at heel strike
(human ~4 deg); rho* = 0.8 tracked only 0.89-0.91x (r9 without the prior: 0.80x).
- human_gait_prior: ratio_tol RATIO_TOL (reward 1 when |rho* - 1| >= tol),
  terminal_knee_weight TERMINAL_KNEE_WEIGHT for gait percent < 8 or > 88.
Starts from r11 last.ckpt. Validate human-likeness on Lencioni (training table is Fukuchi).
"""
from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r11 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r11 import (
    env_config as _r11_env_config, agent_config)  # noqa: F401
from protomotions.envs.context_views import EnvContext

RATIO_TOL = 0.05
TERMINAL_KNEE_WEIGHT = 3.0


def env_config(robot_cfg, args):
    cfg = _r11_env_config(robot_cfg, args)
    prior = cfg.reward_components["human_gait_prior"]
    prior.dynamic_vars["tar_cadence_ratio"] = EnvContext.steering.tar_cadence_ratio
    prior.static_params.update(ratio_tol=RATIO_TOL, terminal_knee_weight=TERMINAL_KNEE_WEIGHT)
    return cfg
