"""vic2_TC_s5_r12 (vic_tc_v2 candidate, 2026-10-28, server A; user: "상체, 상지는 실제 사람 수준 범위의 PD 값으로 설정해줘"):
vic2_TC_s5_r11 (r7 without training pushes) with the trunk, neck / head and wrist PD gains moved to the human range.
Shoulder / elbow were already lowered in r4 (kept; Shoulder_z arm swing 15 -> 10). Sources and the full table:
share/HMV_v1_teacher_261026/trunk_arm_gain_ref.csv (2026-10-28 rows). Axis meaning from human_model_v2.xml: trunk / neck
_x lateral bending, _y flexion-extension, _z axial rotation; Wrist_x flexion, Wrist_y pronation, Wrist_z radial-ulnar.
- Trunk (Torso, Spine, Chest in series, lumped = per joint / 3): lateral bending 160 -> 757 (lumped 53 -> 252; quick
  release 395 +- 241, Cholewicki 2010), axial rotation 107 -> 330 (lumped 36 -> 110; walking 124 +- 54, Prins 2019),
  damping 2.0 / 9.5 / 1.3 -> 75 / 75 / 20 (lumped 25 / 25 / 6.7; sagittal 27.5 +- 9, zeta ~0.5, Moorhouse & Granata 2005;
  axial by the zeta 0.5 rule). Sagittal stiffness 757 (lumped 252) was already human level.
- Neck + Head (2 in series): stiffness kept (passive / effective ranges bracket it; flexion must stay above m g l ~4.4),
  damping 4 / 2.7 / 1.3 -> 1.2 / 1.0 / 0.3 (current zeta 1.4-1.7 is overdamped; head resonates near 2.1 Hz, Peng 1996).
- Wrist: 7 / 6.5 / 7 -> 1.5 / 1.0 / 2.5 N m/rad, damping 0.05 / 0.03 / 0.06 (relaxed 0.4-1.9, Falzarano 2021, Formica 2012).
Scapula (Thorax) and Shoulder_x / y: no human data, kept. Everything else is r11's, so r12 vs r11 isolates the upper body.
"""
from examples.experiments.steering import mlp_human_model_v3_vic2_tc_s5_r4 as _r4
from examples.experiments.steering.mlp_human_model_v3_vic2_tc_s5_r11 import *  # noqa: F401,F403

UPPER_GAINS = {
    **{f"{j}_x": (757.3333333333334, 75.0) for j in ("Torso", "Spine", "Chest")},
    **{f"{j}_y": (757.3333333333334, 75.0) for j in ("Torso", "Spine", "Chest")},
    **{f"{j}_z": (330.0, 20.0) for j in ("Torso", "Spine", "Chest")},
    **{f"{j}_y": (20.0, 1.2) for j in ("Neck", "Head")},
    **{f"{j}_x": (13.333333333333334, 1.0) for j in ("Neck", "Head")},
    **{f"{j}_z": (6.666666666666667, 0.3) for j in ("Neck", "Head")},
    **{f"{s}_Shoulder_z": (10.0, 1.0) for s in "LR"},
    **{f"{s}_Wrist_x": (1.5, 0.05) for s in "LR"},
    **{f"{s}_Wrist_y": (1.0, 0.03) for s in "LR"},
    **{f"{s}_Wrist_z": (2.5, 0.06) for s in "LR"},
}
# Declared plant change (launcher allows exactly these to differ from the frozen robot): r4's arm gains + the above.
PLANT_GAIN_OVERRIDES = {**_r4.PLANT_GAIN_OVERRIDES, **UPPER_GAINS}


def configure_robot_and_simulator(robot_cfg, simulator_cfg, args):
    _r4.configure_robot_and_simulator(robot_cfg, simulator_cfg, args)
    for name, (kp, kd) in UPPER_GAINS.items():
        info = robot_cfg.control.control_info[name]
        info.stiffness = kp
        info.damping = kd
