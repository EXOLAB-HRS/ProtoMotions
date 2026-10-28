"""vic2_TC_s5_r6 (vic_tc_v2 candidate, 2026-10-28, local PC): vic2_TC_s5_r3 (vic2 r1 recipe from vic_TC_s5_r15 last) with
more exploration on the 10 co-activation outputs: their fixed actor log std is set to -1.2 (sigma 0.30) by the launcher
(teacher_followup.py U_LOGSTD) instead of the parent's -2.9 (sigma 0.055). Why: r3 (pushes 0.4 m/s) and r5 (0.6 m/s) kept u
flat after pushes (coact_push_rise 0.001 / -0.002) with u exploration of only ~0.05 around 0, further smoothed by the 50 ms
filter, so the policy never saw what stiffening during a push does. Stronger pushes alone (r5) only added falls. The
recipe itself is r1's (pushes 0.4 m/s); the PD-target and feed-forward outputs keep log std -2.9.
"""
from examples.experiments.steering.mlp_human_model_v3_vic2_tc_s5_r1 import *  # noqa: F401,F403
