"""vic_TC_s5_r15 (implementation vic_tc_v1 on vic_tc_v1_se, candidate, 2026-10-27): strategy A with the teacher se-v1-s5
(s3 + pelvis_gait_prior 0.1 + legs2 discriminator; stopped at 27 % on a plateau) instead of s3.

Why: s3 -> VIC (vic_TC_s3_r12 / r12b / r13) keeps the gait close to human (r13: W3 pass, 5/16 W1-W4 rows fail), but the
remaining failures are ankle curve (r 0.70), cadence, lateral sway and pelvis rows. s5 is the teacher with the best W2 v4
(fails 0/1/3 rows at 0.8/1.0/1.4 m/s vs s3 4/3/4: pelvis obliquity fixed); it fails the switch gate G1 only on
S_walk15 0.875 (< 0.896), which the VIC runs from s3 raised (r13 0.902).
Same as vic_TC_s3_r12 (VIC options, alpha 0 -> 1 then beta 1 -> 0 step schedule with the fall gate, VIC costs) with
the actor lr 2e-5 of r13 (launcher ACTOR_LR and here). Starts from se-v1-s5 last.ckpt with the 20 new actor outputs at
zero; the parent's legs2 discriminator is kept as it is.
"""
import dataclasses

from examples.experiments.steering.mlp_human_model_v3_teacher_se_v1_s5 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_teacher_se_v1_s5 import (
    env_config as _s5_env_config, agent_config as _s5_agent_config)
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.control.vic_steering import (
    VicSteeringConfig, vic_cost, VIC_BAND_COST, VIC_COCONTRACTION, VIC_EFFORT)

VIC_WEIGHTS = {"vic_band": (VIC_BAND_COST, 0.03), "vic_cocontraction": (VIC_COCONTRACTION, 0.03),
               "vic_effort": (VIC_EFFORT, 0.05)}
VIC_OPTIONS = dict(beta_decay=True, fresh_stance=True, damp_delta_gain=0.5, delta_band=True, delta_band_margin=1.5,
                   delta_tau_up_s=0.015, delta_tau_down_s=0.05, frontal_human_ref=True)


def env_config(robot_cfg, args):
    cfg = _s5_env_config(robot_cfg, args)
    old = cfg.control_components["steering"]
    kept = {f.name: getattr(old, f.name) for f in dataclasses.fields(old) if f.name != "_target_"}
    cfg.control_components["steering"] = VicSteeringConfig(**kept, **VIC_OPTIONS)
    for name, (path, weight) in VIC_WEIGHTS.items():
        cfg.reward_components[name] = MdpComponent(
            compute_func=vic_cost, dynamic_vars={"value": path}, static_params={"weight": weight})
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    cfg = _s5_agent_config(robot_cfg, env_cfg, args)
    n = robot_cfg.number_of_actions + 2 * len(VicSteeringConfig.vic_joints)
    cfg.model.actor.num_out = n
    cfg.model.actor.mu_model.num_out = n
    # Must match the launcher's ACTOR_LR (fork_checkpoint writes it into the optimizer state).
    cfg.model.actor_optimizer.lr = 2e-5
    return cfg
