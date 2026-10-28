"""vic2_TC_s5_r14b (vic_tc_v2 candidate, human_model_v3.2, 2026-10-28, server B): vic2_TC_s5_r14 with reflex thresholds
re-calibrated in TRAINING conditions. Why: the r14 smoke showed the reflex active ~25 % of substeps at Hip_x and 1.5-6 % at
the other joints, because r14's thresholds came from deterministic evaluation rollouts. A sensor-only calibration on the
delay-adapted r13b policy in training conditions (256 env, 2M transitions: exploration noise, random commands, resets)
gave high-passed stretch-velocity quantiles well above the evaluation ones. v_th = max(training p99, evaluation walking
p99.9): ~1 % active in training, < 0.1 % in steady evaluation walking; the sagittal joints still exceed it right after the
0.35 m/s lateral push (hip 5.7, knee 11.3, ankle 8.9 rad/s), Hip_x (3.0) no longer does.
"""
from examples.experiments.steering import mlp_human_model_v3_vic2_tc_s5_r14 as _r14
from examples.experiments.steering.mlp_human_model_v3_vic2_tc_s5_r14 import *  # noqa: F401,F403

V_TH = {"Hip_y": 4.7, "Knee_y": 8.4, "Ankle_y": 8.0, "Hip_x": 4.6, "Subtalar_x": 6.5}
_r14.JOINT_REFLEX = dict(_r14.JOINT_REFLEX, v_th=V_TH)
JOINT_REFLEX = _r14.JOINT_REFLEX
