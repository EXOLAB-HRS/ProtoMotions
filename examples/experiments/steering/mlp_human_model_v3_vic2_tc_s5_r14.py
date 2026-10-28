"""vic2_TC_s5_r14 (vic_tc_v2 candidate, 2026-10-28, thresholds calibrated, user decision: joint-level spinal stretch reflex on the plant):
vic2_TC_s5_r13 (= r11, no training pushes, + afferent delay on the actor's body-state observations) with the joint
stretch reflex (protomotions/robot_configs/human_model/common/joint_reflex.py) on the 10 VIC joints.

Why: r13 makes the actor's feedback late (40-70 ms proprio, 30-60 ms vestibular), as in a person. A person also has a
fast spinal path that the cortex does not have to wait for: the short-latency (M1) stretch reflex, ~25-45 ms in leg
muscles, which resists a sudden stretch and is scaled up by descending drive (co-contraction). Without it the
delayed policy is the only answer to a perturbation. r14 adds that path on the plant side, every physics substep:
phasic, velocity-driven, thresholded on the deviation velocity from the PD target (e_v = qdot - qdot_ref) and only
while the deviation grows, delayed d ~ U(30, 40) ms per env, gain g = 1 + kappa * u_+ with the VIC co-activation
level u (kappa 1), clipped at 30 % of the joint effort limit and, with the VIC feed-forward, at the strength caps.
In steady walking it should be silent (v_th above the walking p99), so r14 changes the plant only in reflex situations.
It is the joint-level stand-in for the per-muscle stretch reflex of human-plant-v2, which replaces it later.

Before a smoke / main run (draft):
1. Calibrate V_TH: evaluate the r13 (or r11) checkpoint in steady walking with HC_JOINT_REFLEX_CALIBRATE=<json>
   (sensor only, gain 0) and set V_TH per joint >= the 'away' p99 of |e_v| (2-sided: the values are per joint, both
   antagonist groups); then check that the evaluation pushes exceed it. The values below are uncalibrated placeholders.
2. Training has no pushes (r11 / r13), so the reflex rarely fires in training; its effect shows in the push battery.
3. The launcher (teacher_followup.py) must list "vic2-tc-s5-r14" with r13's entries (ACTOR_LR, U_LOGSTD, VIC2_LEGS_DISC)
   and the run needs --human-pd implicit (the reflex adds to the VIC feed-forward of the implicit drive).
"""
from protomotions.robot_configs.human_model.common.joint_reflex import DEFAULT_KV, JointReflexConfig

from examples.experiments.steering import mlp_human_model_v3_vic2_tc_s5_r13 as _r13
from examples.experiments.steering.mlp_human_model_v3_vic2_tc_s5_r13 import *  # noqa: F401,F403

# Calibrated 2026-10-28 on vic2_TC_s5_r11 (sensor-only block, steady walking 64 env x 25 s, physics rate): the PD-deviation
# signal was as large in walking as after pushes, so the reflex uses signal="highpass" (qdot - lowpass(qdot), 0.1 s) and
# v_th = walking p99.9 of the high-passed stretch velocity, max of left / right (rad/s). Right after the 0.35 m/s lateral
# push the 30 Hz values were hip 5.7, knee 11.3, ankle 8.9, hip_x 3.0 rad/s, i.e. above these thresholds.
V_TH = {"Hip_y": 3.9, "Knee_y": 7.6, "Ankle_y": 7.4, "Hip_x": 2.7, "Subtalar_x": 5.5}
JOINT_REFLEX = dict(enabled=True, delay_ms=(30.0, 40.0), v_th=V_TH, k_v=dict(DEFAULT_KV), kappa=1.0, max_frac=0.3,
                    away_only=True, signal="highpass", hp_tau_s=0.1)


def env_config(robot_cfg, args):
    cfg = _r13.env_config(robot_cfg, args)
    cfg.joint_reflex = JointReflexConfig(**JOINT_REFLEX)
    return cfg
