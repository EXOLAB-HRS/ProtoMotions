"""vic3_TC_s5_r9c (vic_tc_v3 candidate, 2026-10-30, local PC): continuation of vic3_TC_s5_r9b from its last checkpoint for
20M more transitions, same recipe. Started only if the fall termination rate still dropped by more than 20 % over the
last 20 % of r9b's epochs (AGENTS.md training-length rule; user 2026-10-30: "40M이 부족하면 계속 연장").
"""
from examples.experiments.steering.mlp_human_model_v3_vic3_tc_s5_r9b import *  # noqa: F401,F403
