"""vic_TC_b8 (implementation vic_tc_v1 on vic_tc_v1_se, candidate, 2026-10-28): HMV_v1 path B diagnostic run (user-approved,
"1안"). vic_TC_b4 last.ckpt (cb82cccd..., 80M) continued with only the walking-stance Knee_y stiffness reference changed:
k range 15-75 -> 114-185 N m/rad (geometric mean 33.5 -> 145.2). Rewards, commands and every other gain (standing / swing
knee, hip, ankle, frontal, all damping) as b4, i.e. b7 with a narrower knee range.

Why: b7 (81-745, geometric mean 245.7) passed the hip flexion offset at all speeds for the first time in path B and
improved knee loading / double support, but quiet standing fell to 8-9 of 64, pelvis tilted forward 11-13 deg and W1
falls rose. b8 asks whether a lower stance knee reference keeps the gait gain without breaking standing.
Source of 114-185: walking knee weight-acceptance flexion quasi-stiffness of one subject close to the 63.3 kg model
(subject 7, 63.1 kg, 1.45 m, 0.75-2.00 m/s; Shamaei 2014 dissertation tables 3-2 / 3-3, data of Shamaei et al. 2013),
found by Codex 2026-10-28. One subject, one gait phase applied to the whole foot-height stance state, quasi-stiffness not
the local perturbation stiffness k_p, no mass scaling: an exploratory substitute reference, not a measured impedance.
"""
import dataclasses

from examples.experiments.steering.mlp_human_model_v3_vic_tc_b4 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_vic_tc_b4 import env_config as _b4_env_config

KNEE_STANCE_K = ("Knee_y", 1, 114.0, 185.0)


def env_config(robot_cfg, args):
    cfg = _b4_env_config(robot_cfg, args)
    steer = cfg.control_components["steering"]
    cfg.control_components["steering"] = dataclasses.replace(steer, k_range_overrides=(KNEE_STANCE_K,))
    return cfg
