"""teacher_cadence_v1 revision 14: r13 with a usable heel-strike knee signal and a measured pelvis target.

r13 (from r12): knee at heel strike unchanged (22.5 deg, r12 21.9), pelvis 0.946 m, Lencioni knee
r 0.752. knee_strike_prior averaged 0.977 over training: its window (gait % > 90 or < 3, ~13% of
steps) times the |rho* - 1| < 0.05 gate (~22% of envs) left ~3% of samples with any signal.
- knee_strike_prior: no cadence gate (ratio_tol 0), window gait % > LATE_PCT or < EARLY_PCT,
  weight KNEE_STRIKE_WEIGHT;
- pelvis_height_prior: target PELVIS_TARGET = 0.975 m (plant standing) x 0.980 (Lencioni et al.
  2019 walking/standing pelvis-marker height, 29 adults, 0.9-1.3 m/s, SD 0.005).
Starts from r12 last.ckpt (not r13).
"""
from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r13 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r13 import (
    env_config as _r13_env_config, agent_config)  # noqa: F401

KNEE_STRIKE_WEIGHT = 0.30
LATE_PCT = 0.85
EARLY_PCT = 0.05
PELVIS_TARGET = 0.955


def env_config(robot_cfg, args):
    cfg = _r13_env_config(robot_cfg, args)
    knee = cfg.reward_components["knee_strike_prior"]
    knee.static_params.update(ratio_tol=0.0, late_pct=LATE_PCT, early_pct=EARLY_PCT, weight=KNEE_STRIKE_WEIGHT)
    cfg.reward_components["pelvis_height_prior"].static_params.update(target=PELVIS_TARGET)
    return cfg
