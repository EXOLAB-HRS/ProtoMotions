"""vic_TC_r11 (implementation vic_tc_v1, candidate change, 2026-10-26): vic_TC_r6 with the HMV_v1 meeting follow-up
changes to k = k_ref(s) * exp(delta) (share/HMV_v1_teacher_261026/stage_meeting_261026.md, Q2):
  - delta_band: delta_max per joint and leg state = half log-width of the human range + ln 1.5, so ankle stance can
    reach 22-406 N m/rad (1.87 instead of 1.1); Hip_x / Subtalar_x keep 1.1.
  - delta_lowpass_s 0.04: delta lowpassed like muscle activation (r6 changed it every control step).
  - frontal_human_ref: Hip_x / Subtalar_x get scale-level human ranges (HUMAN_RANGES_FRONTAL) instead of the
    current gain as reference.
  - damping coupling g = 0.5 (damping ratio held) as r4 / r6.
Starts from r1 last.ckpt like r2-r10, so r11 vs r6 shows the effect of these three changes together.
"""
import dataclasses

from examples.experiments.steering.mlp_human_model_v3_vic_tc_r6 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_vic_tc_r6 import (
    env_config as _r6_env_config, agent_config)  # noqa: F401

DELTA_BAND_MARGIN = 1.5
DELTA_LOWPASS_S = 0.04


def env_config(robot_cfg, args):
    cfg = _r6_env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(
        cfg.control_components["steering"], delta_band=True, delta_band_margin=DELTA_BAND_MARGIN,
        delta_lowpass_s=DELTA_LOWPASS_S, frontal_human_ref=True)
    return cfg
