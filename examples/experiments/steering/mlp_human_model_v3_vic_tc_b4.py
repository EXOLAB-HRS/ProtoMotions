"""vic_TC_b4 (implementation vic_tc_v1 on vic_tc_v1_se, candidate, 2026-10-27): HMV_v1 path B human-gain teacher
started from se-v1-s3 last.ckpt (1b77d621...) instead of r25 (b3 walked a fast shuffle at the human gain and could
not stand still at 0 m/s; s3 stands 58/64 and fails W2 v4 4|3|4 at the old gain).

- Rewards, commands, inputs: teacher_se_v1_s3 (as vic_TC_s3_r12), with the commanded natural cadence set to the
  human law of the W2 judge, f0 = 0.558 v + 1.060 steps/s (vic_TC_s3_r16 showed the pack law alone fails cadence).
- VIC as vic_TC_b2: alpha = 1, beta = 0, delta = 0 from the first step (no schedule, no delta band or filter),
  frontal_human_ref, fresh_stance, damping law 0.5; effort cost 0.05, no band / co-contraction cost.
- Differs from the strategy A line (vic_TC_s3_r12 -> r13 -> r16) only in skipping the alpha / beta schedule.
Actor lr 2e-5 (launcher ACTOR_LR, as r13-r17).
"""
import dataclasses

from examples.experiments.steering.mlp_human_model_v3_teacher_se_v1_s3 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_teacher_se_v1_s3 import (
    env_config as _s3_env_config, agent_config as _s3_agent_config)
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.control.vic_steering import VicSteeringConfig, vic_cost, VIC_EFFORT

VIC_OPTIONS = dict(alpha_init=1.0, beta_init=0.0, schedule=False, delta_max=0.0, delta_band=False,
                   delta_lowpass_s=0.0, fresh_stance=True, damp_delta_gain=0.5, frontal_human_ref=True)
HUMAN_CADENCE = dict(cadence_f0_slope=0.558, cadence_f0_intercept=1.060)


def env_config(robot_cfg, args):
    cfg = _s3_env_config(robot_cfg, args)
    old = cfg.control_components["steering"]
    kept = {f.name: getattr(old, f.name) for f in dataclasses.fields(old)
            if f.name != "_target_" and f.name not in HUMAN_CADENCE}
    cfg.control_components["steering"] = VicSteeringConfig(**kept, **VIC_OPTIONS, **HUMAN_CADENCE)
    cfg.reward_components["vic_effort"] = MdpComponent(
        compute_func=vic_cost, dynamic_vars={"value": VIC_EFFORT}, static_params={"weight": 0.05})
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    cfg = _s3_agent_config(robot_cfg, env_cfg, args)
    n = robot_cfg.number_of_actions + 2 * len(VicSteeringConfig.vic_joints)
    cfg.model.actor.num_out = n
    cfg.model.actor.mu_model.num_out = n
    # Must match the launcher's ACTOR_LR (fork_checkpoint writes it into the optimizer state).
    cfg.model.actor_optimizer.lr = 2e-5
    return cfg
