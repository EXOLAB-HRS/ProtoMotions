"""teacher_se_v1 s5 (2026-10-26, candidate run recipe): s4pelvis (s3 + pelvis_gait_prior 0.1) with the s5amp_legs2
discriminator (pelvis + legs + trunk bodies, two most recent history steps), 400M from se-v1-s3 last.ckpt.

AMP screens (25M each from s3, same seed / env / batch; control = the first 25M of s4pelvis):
  control (all 26 bodies, 8 steps): agent_acc 0.983, pos_acc 1.000, amp reward 0.029
  legs (14 bodies, 8 steps):        agent_acc 0.984, pos_acc 1.000, amp reward 0.044
  legs2 (14 bodies, 2 steps):       agent_acc 0.92 -> 0.986, pos_acc 0.999, amp reward 0.065
Neither removes the saturation (the discriminator re-separates within ~100 epochs), but legs2 doubles the style reward
with task terms and falls unchanged, so s5 keeps it. Watched hourly with scripts/train_watch.py; stopped early if the
pelvis and gait terms plateau.
"""
from examples.experiments.steering import mlp_human_model_v3_teacher_se_v1_s5amp as _base
from examples.experiments.steering.mlp_human_model_v3_teacher_se_v1_s5amp import *  # noqa: F401,F403

_base.DISC_BODIES = "legs"
_base.DISC_HISTORY = 2
