"""vic2_TC_s5_r4 (vic_tc_v2 candidate, 2026-10-28; user approved the arm-gain change): vic2_TC_s5_r1 with the arm PD gains
lowered toward the human relaxed / posture range. Why: vic_TC_s5_r15 walks with asymmetric, odd arm motion; its arm gains
are well above a walking person's (share/HMV_v1_teacher_261026/trunk_arm_gain_ref.csv: shoulder 24-54 vs relaxed 2-10,
posture 15-30 N m/rad; elbow flexion 35 vs relaxed 2-10, up to ~40 only with activity). New values (kd = 0.1 kp as the
model's rule): Shoulder_x 15, Shoulder_y 10, Shoulder_z 15, Elbow_z 8 N m/rad, both sides. Effort limits, Thorax, Wrist,
trunk and the passive Elbow_x/y / Hand joints are unchanged. Everything else is r1's (per-joint co-activation level,
pushes, symmetry reward, upper arms in the discriminator).
"""
from examples.experiments.steering import mlp_human_model_v3_vic2_tc_s5_r1 as _r1
from examples.experiments.steering.mlp_human_model_v3_vic2_tc_s5_r1 import *  # noqa: F401,F403

ARM_KP = {"Shoulder_x": 15.0, "Shoulder_y": 10.0, "Shoulder_z": 15.0, "Elbow_z": 8.0}
# Declared plant change: the launcher allows exactly these gains to differ from the frozen robot (teacher_followup.py).
PLANT_GAIN_OVERRIDES = {f"{side}_{j}": (kp, 0.1 * kp) for side in "LR" for j, kp in ARM_KP.items()}


def configure_robot_and_simulator(robot_cfg, simulator_cfg, args):
    _r1.configure_robot_and_simulator(robot_cfg, simulator_cfg, args)
    for name, (kp, kd) in PLANT_GAIN_OVERRIDES.items():
        info = robot_cfg.control.control_info[name]
        info.stiffness = kp
        info.damping = kd
