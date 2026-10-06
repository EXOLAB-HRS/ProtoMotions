"""teacher_cadence_v1 revision 4: the r3 recipe unchanged; only the warm start differs.

r3 (261003 e01) learned no rho* response: after 20M steps the critic's cadence/clock input
weights were 2-6x the parent's column average, the actor's 1/6-1/14. r4 starts the actor's
three new columns from random weights with the parent's column std
(teacher_followup.py --append-cadence-input, experiment teacher-cadence-v1-r4).
"""
from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r3 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r3 import env_config, agent_config  # noqa: F401
