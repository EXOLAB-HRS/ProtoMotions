"""vic_TC_b3 (implementation vic_tc_v1, candidate, 2026-10-27): HMV_v1 path B human-gain teacher, the vic_TC_b2 recipe
started from teacher_cadence_v1 r25 last.ckpt (launcher --parent) instead of scratch (user, after b2 did not converge
in 300M). alpha 1, beta 0, delta 0 from the first step, no alpha schedule (Codex review, 2026-10-27).
"""
from examples.experiments.steering.mlp_human_model_v3_vic_tc_b2 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_vic_tc_b2 import env_config, agent_config  # noqa: F401
