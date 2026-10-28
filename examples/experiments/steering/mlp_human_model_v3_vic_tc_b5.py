"""vic_TC_b5 (implementation vic_tc_v1 on vic_tc_v1_se, candidate, 2026-10-28): HMV_v1 path B, vic_TC_b4 continued from
its last.ckpt (cb82cccd..., 80M, epoch 608) with the existing pelvis gait prior of teacher_se_v1_s4pelvis added.

Why: b4 80M fails HMV-Walk; quiet standing (38-40/64) and cadence pass at all three speeds, but W2 v4 keeps failing
pelvis_tilt_curve / pelvis_tilt_events / pelvis_obliquity_curve (50M -> 80M fails 6|5|7 -> 6|8|8, shape stalled).
human_gait_prior (0.25) scores hip / knee / ankle only. Codex review 2026-10-28: add only pelvis_gait_prior, keep
everything else (alpha 1, beta 0, delta 0, frontal_human_ref, k_ref, tau_res 0.33, cadence law, AMP, actor lr 2e-5).
The pelvis term does not directly target hip offset, double support or symmetry.
The pelvis table is built from the WBDS rows the W2 v4 judge uses, so WBDS rows are not an independent check (report VC).
"""
from protomotions.envs.context_views import EnvContext
from protomotions.envs.control.cadence_gait_prior import GAIT_PCT, GAIT_PCT_VALID, pelvis_gait_prior
from protomotions.envs.mdp_component import MdpComponent

from examples.experiments.steering.mlp_human_model_v3_vic_tc_b4 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_vic_tc_b4 import env_config as _b4_env_config
from examples.experiments.steering.mlp_human_model_v3_teacher_se_v1_s4pelvis import (
    PELVIS_WEIGHT, TILT_RATE_SD_FLOOR, KERNEL_SCALE, PELVIS_TABLE)


def env_config(robot_cfg, args):
    cfg = _b4_env_config(robot_cfg, args)
    cfg.reward_components["pelvis_gait_prior"] = MdpComponent(
        compute_func=pelvis_gait_prior,
        dynamic_vars={"root_rot": EnvContext.current.root_rot, "root_ang_vel": EnvContext.current.root_ang_vel,
                      "tar_speed": EnvContext.steering.tar_speed, "gait_pct": GAIT_PCT, "gait_pct_valid": GAIT_PCT_VALID,
                      "tar_cadence_ratio": EnvContext.steering.tar_cadence_ratio},
        static_params={"table_file": str(PELVIS_TABLE), "tilt_rate_sd_floor": TILT_RATE_SD_FLOOR,
                       "obl_sd_floor_deg": 1.0, "min_speed": 0.3, "kernel_scale": KERNEL_SCALE, "ratio_tol": 0.3, "weight": PELVIS_WEIGHT})
    return cfg
