"""vic4_TC_s5_from-ffdv33r2e453_band1-w33_r1 (strategy B stage 2, jinsu local, 2026-11-03; user: "알아서 돌려라", handoff
261103 section 8 candidate "강성 범위 폭을 줄여서 열기"): from vic4c_TC_s5_from-ffdistillv33r1_r2 epoch_453 (plant v3.3, rank 1),
the stage-2 recipe vic4_TC_s5_from-ffdv33r2e453_unfreeze_r1 (stiffness outputs unfrozen, alpha 1 / beta 0, feed-forward
distillation, swing knee out of r_imp) with one law change and the handoff's distillation teacher:

- v4_band_margin 2.0 -> 1.0: ln k_tot = ln k_c(phi) + (h(phi) + ln margin) * a, so a = +-1 reaches the ends of the human
  range instead of twice beyond them. Reason: when the stiffness was opened at margin 2 on the server, 43-68 % of the knee
  k and b left the human range and the 1.4 m/s knee collapsed (r 0.62-0.68); the frozen centre (e453) held.
- Distillation teacher: env HC_FF_DISTILL_TEACHER = walk_TC_s5_from-s5-v33_r1 last (v3.3 fixed-PD walker, 437d7f66...),
  the handoff's recommended v3.3 teacher (e453 and unfreeze_r1 used se-v1-s5 on v3.1).
"""
import dataclasses

from examples.experiments.steering import mlp_human_model_v3_vic4_tc_s5_from_ffdv33r2e453_unfreeze_r1 as _u
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_s5_from_ffdv33r2e453_unfreeze_r1 import *  # noqa: F401,F403

V4_BAND_MARGIN = 1.0


def env_config(robot_cfg, args):
    cfg = _u.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(cfg.control_components["steering"], v4_band_margin=V4_BAND_MARGIN)
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _u.agent_config(robot_cfg, env_cfg, args)
