"""vic2_TC_s5_r3 (vic_tc_v2 candidate, 2026-10-28, local PC): vic2_TC_s5_r1's recipe started from vic_TC_s5_r15 last.ckpt
(on Drive, sha b2e034db) instead of the server-only r17 epoch 190. The launcher --parent decides the start; the recipe is
r1's (the cadence law and VIC state file come from the r17 chain)."""
from examples.experiments.steering.mlp_human_model_v3_vic2_tc_s5_r1 import *  # noqa: F401,F403
