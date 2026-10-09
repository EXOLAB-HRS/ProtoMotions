"""teacher_cadence_v1 revision 15: r12 rewards and settings unchanged; AMP motion pack from human lab data.

r12-r14 landed at ~22 deg knee flexion (human ~3-6 deg). With one knee definition (hip centre, lateral knee,
lateral ankle, standing = 0) the AMASS source clips land at 17.9 deg, the forward25 pack at 18.0 deg and r12
at 21.7 deg, against 5.9 deg for Lencioni et al. 2019 markers: the policy copies its AMP reference.
r15 trains with --motion = the Lencioni 2019 pack (scripts/pack_from_lencioni.py: leg and pelvis angles
from adult walking trials, subjects with id % 5 == 0 held out; trunk and arms from the forward25 mean cycle).
Starts from r12 last.ckpt.
"""
from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r12 import *  # noqa: F401,F403

# The conditional AMP discriminator's cadence labels are per pack frame; r3 reads its label file from a
# module global at call time, so r15 points it at the labels built for the Lencioni pack.
from pathlib import Path as _Path
import examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r3 as _r3

_r3.R3_LABEL_FILE = _Path(__file__).resolve().parents[4] / "share/cadence/lencioni65_13941029_cadence_labels_r080_125.pt"
import examples.experiments.steering.mlp_human_model_v3_c_a7_cadence as _c_a7_cadence

_c_a7_cadence.LABEL_FILE = _Path(__file__).resolve().parents[4] / "share/cadence/lencioni65_13941029_cadence_labels.pt"
