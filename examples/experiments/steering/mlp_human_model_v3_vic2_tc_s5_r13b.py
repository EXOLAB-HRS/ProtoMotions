"""vic2_TC_s5_r13b (vic_tc_v2 candidate, 2026-10-28, server B): continuation of vic2_TC_s5_r13 (r11 + afferent delay) from its
last checkpoint for 20M more transitions, same recipe. Why (AGENTS.md training-length rule, 2026-10-28): in r13's last 20 % of
epochs the fall termination rate still dropped 33 % (0.00079 -> 0.00053; r11 ~0.0003) and episode length still rose, so the
policy was still adapting to the delay at 40M.
"""
from examples.experiments.steering.mlp_human_model_v3_vic2_tc_s5_r13 import *  # noqa: F401,F403
