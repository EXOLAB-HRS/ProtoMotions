"""vic2_TC_s5_r2 (vic_tc_v2 candidate, 2026-10-28): vic2_TC_s5_r1 without training pushes (control). Does the per-leg
co-activation level learn to rise after disturbances only when the training contains disturbances?"""
from examples.experiments.steering import mlp_human_model_v3_vic2_tc_s5_r1 as _r1
from examples.experiments.steering.mlp_human_model_v3_vic2_tc_s5_r1 import *  # noqa: F401,F403

_r1.PUSH = False
