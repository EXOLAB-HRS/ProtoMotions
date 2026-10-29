"""vic_TC_s5_r18 (implementation vic_tc_v1 on vic_tc_v1_se, candidate, 2026-10-28): vic_TC_s5_r17's setting (r15 continued at
alpha 1, beta 0, actor lr 2e-5, human cadence law 0.558 v + 1.060) for 30M only. In r17 the W4 mechanical-work ratio fell with
training (epoch 190 / ~25M 0.53 pass -> epoch 380 0.36 -> last 0.19) while cadence was fixed early, so a short run with
every saved epoch evaluated should keep W4 and the cadence fix. Same recipe as r17; the length is the launcher --steps.
"""
from examples.experiments.steering.mlp_human_model_v3_vic_tc_s5_r17 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_vic_tc_s5_r17 import env_config, agent_config  # noqa: F401
