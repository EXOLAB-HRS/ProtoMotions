"""vic4_TC_s5_from-s5_ffdistill-v33-kaff_r1 (strategy B, jinsu, server, 2026-11-03; user: "추천대로 고고"): the cleaned-law recipe
vic4_TC_s5_from-s5_ffdistill_r1 of slot B on plant human_model_v3.3 with one change: fixed log std of the knee and ankle
feed-forward outputs -2.9 -> ln 0.2 (sigma 0.055 -> 0.2). Hip and other feed-forward outputs stay at sigma 0.055. Reason: the
feed-forward head grows slowly (1/25 of the PD-target head, no cap saturation), while opening all ten outputs at sigma 0.30
(ffx) turned hip noise into shuffling (0.18-0.30 m/s). Output rows are 69 + vic_joints order: L_Knee_y 70, L_Ankle_y 71,
R_Knee_y 73, R_Ankle_y 74. Actor lr stays 2e-5, tau_res_scale stays as in the parent recipe.
"""
import math

from examples.experiments.steering import mlp_human_model_v3_vic4_tc_s5_from_s5_ffdistill_r1 as _ffd
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_s5_from_s5_ffdistill_r1 import *  # noqa: F401,F403

KNEE_ANKLE_FF_LOGSTD = math.log(0.2)
KNEE_ANKLE_FF_SLICES = [[70, 72, KNEE_ANKLE_FF_LOGSTD], [73, 75, KNEE_ANKLE_FF_LOGSTD]]


def agent_config(robot_cfg, env_cfg, args):
    cfg = _ffd.agent_config(robot_cfg, env_cfg, args)
    cfg.model.actor.logstd_slices = list(cfg.model.actor.logstd_slices or []) + KNEE_ANKLE_FF_SLICES
    return cfg
