"""vic_TC_scratch_r16 (implementation vic_tc_v1 on vic_tc_v1_se, candidate, 2026-10-27, hyojun): strategy C stage 1,
variable joint impedance from scratch with no fixed-PD teacher step (share/HMV_v1_teacher_261026/stage_meeting_261026.md,
section 5: "C: k_ref(s) 주변 delta를 처음부터 학습").

Question: is the fixed-PD teacher step needed at all? Compared with strategy A (vic_TC_s3_r12/r13: se-v1-s3 -> VIC)
under the same HMV-Exo judge.

- Rewards, inputs, commands: teacher_se_v1_s1 (the from-scratch first stage of se-v1: SE_v1 inputs, 0.8-1.2 m/s,
  no stops or turns, natural cadence), so C differs from the se-v1 scratch run only by VIC.
- VIC as vic_TC_s3_r12 (same 10 joints, costs 0.03 / 0.03 / 0.05, fresh_stance, damping law 0.5, r11 options) but
  alpha = 1, beta = 0 held from the first step (no schedule, no parent): the drive is at the human reference
  k_ref(s) (HMV-ImpRef_v1 geometric centres) and the policy only learns delta and the feed-forward torque.
- Actor lr 2e-5 as the se-v1 scratch runs (launcher --fresh, no fork).
Stage 2: vic_TC_scratch_r17 (s3 commands) continues from this run's last.ckpt.
"""
import dataclasses

from examples.experiments.steering.mlp_human_model_v3_teacher_se_v1_s1 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_teacher_se_v1_s1 import (
    env_config as _s1_env_config, agent_config as _s1_agent_config)
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.control.vic_steering import (
    VicSteeringConfig, vic_cost, VIC_BAND_COST, VIC_COCONTRACTION, VIC_EFFORT)

VIC_WEIGHTS = {"vic_band": (VIC_BAND_COST, 0.03), "vic_cocontraction": (VIC_COCONTRACTION, 0.03),
               "vic_effort": (VIC_EFFORT, 0.05)}
VIC_OPTIONS = dict(alpha_init=1.0, beta_init=0.0, schedule=False, beta_decay=False, fresh_stance=True,
                   damp_delta_gain=0.5, delta_band=True, delta_band_margin=1.5,
                   delta_tau_up_s=0.015, delta_tau_down_s=0.05, frontal_human_ref=True)
ACTOR_LR = 2e-5


def add_vic(cfg):
    old = cfg.control_components["steering"]
    kept = {f.name: getattr(old, f.name) for f in dataclasses.fields(old) if f.name != "_target_"}
    cfg.control_components["steering"] = VicSteeringConfig(**kept, **VIC_OPTIONS)
    for name, (path, weight) in VIC_WEIGHTS.items():
        cfg.reward_components[name] = MdpComponent(
            compute_func=vic_cost, dynamic_vars={"value": path}, static_params={"weight": weight})
    return cfg


def widen_actor(cfg, robot_cfg):
    n = robot_cfg.number_of_actions + 2 * len(VicSteeringConfig.vic_joints)
    cfg.model.actor.num_out = n
    cfg.model.actor.mu_model.num_out = n
    # Must match the launcher's ACTOR_LR when a stage starts from a parent (fork_checkpoint writes it).
    cfg.model.actor_optimizer.lr = ACTOR_LR
    return cfg


def env_config(robot_cfg, args):
    return add_vic(_s1_env_config(robot_cfg, args))


def agent_config(robot_cfg, env_cfg, args):
    return widen_actor(_s1_agent_config(robot_cfg, env_cfg, args), robot_cfg)
