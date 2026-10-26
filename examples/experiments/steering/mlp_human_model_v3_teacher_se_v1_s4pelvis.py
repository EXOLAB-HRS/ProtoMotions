"""teacher_se_v1 stage 4 with a pelvis gait prior (2026-10-26, user-approved).

s3 (steady speed term 0.15, cadence floor 0.3 m/s) plus:
- pelvis_gait_prior (weight PELVIS_WEIGHT): pelvis tilt rate and obliquity close to WBDS (Fukuchi 2018) by speed and gait
  percent (cadence_gait_prior.pelvis_gait_prior, table share/human_ref/wbds2018_pelvis_tiltrate_obliquity_by_speed_w2v4.pt
  from scripts/build_pelvis_gait_table.py), on the W2 v4 judge's WBDS rows and conventions: tilt rate = d(tilt)/dt in
  deg/s (stride from the cadence law f* = 0.558 v + 1.060 steps/s), obliquity ipsilateral side up + and cycle-mean removed,
  each leg at its own gait percent. Same gates as s3's human_gait_prior (gait_pct_valid, ratio_tol 0.3) and
  tar_speed >= 0.3 m/s (s3's cadence floor).
Why: s3 (teacher-se-v1-s3_v31_parent_seed2944_ghlee/last.ckpt) passes W1, but W2 v4 fails almost only on pelvis rows at
0.8 / 1.0 / 1.4 m/s: pelvis_tilt_curve (VC mean-removed 0.47-0.53 of the cycle inside +-2 SD), pelvis_tilt_events (tilt
ROM z = 1.9-3.0 WBDS, 2.8-4.4 VC) and pelvis_obliquity_curve (0.59-0.68). human_gait_prior scores only hip / knee / ankle.
The tilt offset already passes (pelvis_tilt_offset z 1.1-1.3), so the tilt is scored as a rate (offset-free: shape and
ROM). A first version scored standing-referenced tilt; its WBDS SD (~5 deg, mostly subject offsets) hid the ROM.
TILT_RATE_SD_FLOOR (deg/s): the smallest human rate SD in the table (3.8 deg/s at 0.4 m/s; 5.7 at 1.0 m/s) rounded up,
so no turning point of the human curve gets a near-zero SD.
KERNEL_SCALE 2 (error divided by 2 x SD, as human_gait_prior's kernel_scale; s3 uses 3 there): at 1 the term saturated on
s3 rollouts (tilt-rate z^2 ~10 per step, median reward 0.06-0.16, ~80 % of steps below 0.5).
"""
from pathlib import Path

from protomotions.envs.context_views import EnvContext
from protomotions.envs.control.cadence_gait_prior import GAIT_PCT, GAIT_PCT_VALID, pelvis_gait_prior
from protomotions.envs.mdp_component import MdpComponent

from examples.experiments.steering.mlp_human_model_v3_teacher_se_v1_s3 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_teacher_se_v1_s3 import env_config as _s3_env_config

PELVIS_WEIGHT = 0.1
TILT_RATE_SD_FLOOR = 4.0
KERNEL_SCALE = 2.0
PELVIS_TABLE = Path(__file__).resolve().parents[4] / "share/human_ref/wbds2018_pelvis_tiltrate_obliquity_by_speed_w2v4.pt"


def env_config(robot_cfg, args):
    cfg = _s3_env_config(robot_cfg, args)
    cfg.reward_components["pelvis_gait_prior"] = MdpComponent(
        compute_func=pelvis_gait_prior,
        dynamic_vars={"root_rot": EnvContext.current.root_rot, "root_ang_vel": EnvContext.current.root_ang_vel,
                      "tar_speed": EnvContext.steering.tar_speed, "gait_pct": GAIT_PCT, "gait_pct_valid": GAIT_PCT_VALID,
                      "tar_cadence_ratio": EnvContext.steering.tar_cadence_ratio},
        static_params={"table_file": str(PELVIS_TABLE), "tilt_rate_sd_floor": TILT_RATE_SD_FLOOR,
                       "obl_sd_floor_deg": 1.0, "min_speed": 0.3, "kernel_scale": KERNEL_SCALE, "ratio_tol": 0.3, "weight": PELVIS_WEIGHT})
    return cfg
