"""vic2_TC_s5_r11 (vic_tc_v2 candidate, 2026-10-28, server B): vic2_TC_s5_r7 without the training pushes. Why: every v2 run
with pushes (r1, r4, r7, r8; local r3, r5, r6, r9) lost command tracking against the parent vic_TC_s5_r17 epoch 190
(falls 0.32-0.84 / min, speed error 0.12-0.21, lateral 0.11-0.12), and none of them raised u after a push
(coact_push_rise <= 0.002), so the pushes cost tracking without buying stiffening. The no-push control r2 kept tracking
but also had symmetry and the arm discriminator, which r8 vs r7 shows cost the leg curves (knee r 0.96 -> 0.91, ankle r
0.76 -> 0.68). r11 tells whether the v2 controller alone (per-joint u inside the human range, 50/150 ms filter, cost on
u, u log std -1.2, r4 arm gains) keeps the parent's walking; it is the base for a later run with a reason to stiffen.
"""
from examples.experiments.steering import mlp_human_model_v3_vic2_tc_s5_r1 as _r1
from examples.experiments.steering.mlp_human_model_v3_vic2_tc_s5_r7 import *  # noqa: F401,F403

_r1.PUSH = False
