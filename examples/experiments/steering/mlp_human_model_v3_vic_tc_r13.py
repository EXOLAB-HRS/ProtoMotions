"""vic_TC_r13 (implementation vic_tc_v1 on vic_tc_v1_se, candidate, 2026-10-27): vic_TC_r12b with the actor learning rate
5e-5 -> 2e-5 (the teacher_se_v1 s-series value; set in the launcher's ACTOR_LR, not here). r12 ended with clip_frac 0.47
and r12b starts at 0.63 with actor updates skipped (clip_frac > 0.6), against 0.29-0.30 for the s3/s5 teachers at 2e-5.
Same parent (r12 last.ckpt), same resumed alpha/beta, same seed as r12b, so r12b vs r13 isolates the step size.
"""
from examples.experiments.steering.mlp_human_model_v3_vic_tc_r12b import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_vic_tc_r12b import env_config, agent_config  # noqa: F401
