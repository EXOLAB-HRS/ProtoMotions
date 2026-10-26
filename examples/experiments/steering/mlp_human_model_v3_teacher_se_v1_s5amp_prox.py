"""teacher_se_v1 s5amp_prox: s5amp with DISC_BODIES = "prox" (see mlp_human_model_v3_teacher_se_v1_s5amp.py)."""
from examples.experiments.steering import mlp_human_model_v3_teacher_se_v1_s5amp as _base
from examples.experiments.steering.mlp_human_model_v3_teacher_se_v1_s5amp import *  # noqa: F401,F403

_base.DISC_BODIES = "prox"
