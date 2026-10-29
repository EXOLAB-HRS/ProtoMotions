"""vic_TC_b7 (implementation vic_tc_v1 on vic_tc_v1_se, candidate, 2026-10-28): HMV_v1 path B diagnostic run outside the
b3-b6 cap (user-approved). vic_TC_b4 last.ckpt (cb82cccd..., 80M) continued with only the walking-stance Knee_y stiffness
reference changed: k range 15-75 -> 81-745 N m/rad (geometric mean 33.5 -> 245.7). Rewards, commands and every other
gain (standing / swing knee, hip, ankle, frontal, all damping) as b4; no pelvis prior or cadence 0.3 (b5 / b6).

Why: b3-b6 all fail the knee loading / swing peak, hip offset, double support and symmetry rows. The current stance knee
range is a relaxed-static value (no walking stance measurement). In b4 80M walking the knee PD torque is about equal to
the residual torque (0.94-1.19 : 1), hip PD dominates (25-31 : 1), and the residual never reaches 95 % of its cap
(Codex review 2026-10-28, ~36k env-steps per speed).
Source of 81-745: knee quasi-stiffness in walking, Shamaei et al. 2013 (PLoS ONE e59993; 14 subjects, 0.75-2.63 m/s,
full extension-to-flexion range). Quasi-stiffness is the slope of the walking moment-angle curve, not the local
perturbation stiffness k_p, and the range is not normalised to 63.3 kg: this is a substitute reference for a diagnosis,
not a measured walking impedance or an optimum.
"""
import dataclasses

from examples.experiments.steering.mlp_human_model_v3_vic_tc_b4 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_vic_tc_b4 import env_config as _b4_env_config

KNEE_STANCE_K = ("Knee_y", 1, 81.0, 745.0)


def env_config(robot_cfg, args):
    cfg = _b4_env_config(robot_cfg, args)
    steer = cfg.control_components["steering"]
    cfg.control_components["steering"] = dataclasses.replace(steer, k_range_overrides=(KNEE_STANCE_K,))
    return cfg
