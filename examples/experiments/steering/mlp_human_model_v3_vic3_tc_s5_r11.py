"""vic3_TC_s5_r11 (vic_tc_v3 candidate, 2026-10-30, local PC): continuation of vic3_TC_s5_r7 (best so far: teacher_score_v2
46, W2 pass) from its last checkpoint for 40M more transitions, same recipe. Targets the falls row (0.32/min vs 0.05);
only the start point and length change (user rule 2026-10-28: recipe weight / length / start changes may proceed). The
parent already has the 237 efference input columns, so the launcher does not append them again.
"""
from examples.experiments.steering.mlp_human_model_v3_vic3_tc_s5_r7 import *  # noqa: F401,F403
