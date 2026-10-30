"""walk_TC_scratch_r1 (2026-10-30, user decision "걷기 먼저 → 고정 → VIC"; scratch strategy A stage A1a): the walking
teacher from scratch with the high-gain fixed PD of the nature-conference teacher pc_240_4_r3 (identical gains to
teacher_se_v1 s5), SE_v1 teacher inputs, s5 rewards (pelvis_gait_prior 0.1, steady speed term) and the s5 legs2
discriminator. Stage 1 of a designed two-stage curriculum, as teacher_se_v1 s1 -> s2 (not ad-hoc fine-tune stacking):
narrow commands 0.8-1.2 m/s, no stops, no turns, natural cadence (rho* = 1), 150M. Stage 2 is
walk_TC_scratch_from-walkr1_r2 (full s5 command set, 350M). No sensory delay (teacher = true state, SE_v1).
Plant v3.1 as s5: the joint reflex torque of human_model_v3.2 is applied only through the VIC feed-forward path
(dynamics._vic_feedforward), so with fixed PD it would be a sensor only; the reflex joins at the VIC stage (A2).
New-run defaults (2026-10-30): AMP replay fix and desynced first timeouts (agent / AMP configs), ckpt 10 % (launcher).
"""
from examples.experiments.steering.mlp_human_model_v3_teacher_se_v1_s5 import *  # noqa: F401,F403
from examples.experiments.steering import mlp_human_model_v3_teacher_se_v1_s5 as _s5


def env_config(robot_cfg, args):
    cfg = _s5.env_config(robot_cfg, args)
    s = cfg.control_components["steering"]           # = teacher_se_v1_s1 command set
    s.tar_speed_min, s.tar_speed_max = 0.8, 1.2
    s.stop_probability = 0.0
    s.turn_fraction = 0.0
    s.speed_anchor_targets, s.speed_anchor_probability = (0.8, 1.0, 1.2), 0.5
    s.cadence_natural_fraction = 1.0
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _s5.agent_config(robot_cfg, env_cfg, args)
