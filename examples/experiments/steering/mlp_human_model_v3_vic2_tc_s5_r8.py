"""vic2_TC_s5_r8 (vic_tc_v2 candidate, 2026-10-28, server B): vic2_TC_s5_r7 + the left-right gait symmetry reward 0.2
(arms still out of the discriminator). r7 vs r8 separates the symmetry reward from the arm discriminator as the cause of
the collapsed leg curves in v2 runs (r2: knee r 0.40, ankle r 0.20)."""
from examples.experiments.steering import mlp_human_model_v3_vic2_tc_s5_r1 as _r1
from examples.experiments.steering.mlp_human_model_v3_vic2_tc_s5_r7 import *  # noqa: F401,F403

_r1.SYM_WEIGHT = 0.2
