"""vic_TC_r12 (implementation vic_tc_v1 on the integration branch vic_tc_v1_se, candidate, 2026-10-26): strategy A
step 2, the best fixed-PD teacher se-v1-s3 converted to variable joint impedance.

Question: does a better teacher convert better than r25 (vic_TC_r1)? s3 passes the VIC switch gate (W1 v2, quiet
standing 58/64, hip/knee/ankle curves 9/9; html/vic_TC_261026_strategyA_gate_results_plan_ghlee.html), r25 also
passes it and became r1.

- Rewards, commands, inputs: teacher_se_v1_s3 (r25 rewards, SE_v1 inputs, reset_speed_from_spawn, steady speed term
  0.15, cadence lower bound 0.3 m/s). Heading weight unchanged (s3 already tracks speed; r2's x1.2 fixed r1's error).
- VIC as vic_TC_r1: 10 joints, alpha 0 -> 1 then beta 1 -> 0 on the step schedule with the fall gate; band,
  co-contraction and effort costs (0.03 / 0.03 / 0.05); no metabolic cost (as r6).
- vic_TC_r2 start-up fix (fresh_stance) and r4 damping law (damp_delta_gain 0.5).
- vic_TC_r11 options: delta_band (delta_max per joint and leg state from the human range + ln 1.5),
  delta read as co-contraction and filtered like muscle activation (up 15 ms / down 50 ms), frontal_human_ref
  (Hip_x / Subtalar_x scale-level human ranges).
Starts from se-v1-s3 last.ckpt (1b77d621...) with the 20 new actor outputs at zero (identical actions at alpha = 0).
"""
import dataclasses

from examples.experiments.steering.mlp_human_model_v3_teacher_se_v1_s3 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_teacher_se_v1_s3 import (
    env_config as _s3_env_config, agent_config as _s3_agent_config)
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.control.vic_steering import (
    VicSteeringConfig, vic_cost, VIC_BAND_COST, VIC_COCONTRACTION, VIC_EFFORT)

VIC_WEIGHTS = {"vic_band": (VIC_BAND_COST, 0.03), "vic_cocontraction": (VIC_COCONTRACTION, 0.03),
               "vic_effort": (VIC_EFFORT, 0.05)}
VIC_OPTIONS = dict(beta_decay=True, fresh_stance=True, damp_delta_gain=0.5, delta_band=True, delta_band_margin=1.5,
                   delta_tau_up_s=0.015, delta_tau_down_s=0.05, frontal_human_ref=True)


def env_config(robot_cfg, args):
    cfg = _s3_env_config(robot_cfg, args)
    old = cfg.control_components["steering"]
    kept = {f.name: getattr(old, f.name) for f in dataclasses.fields(old) if f.name != "_target_"}
    cfg.control_components["steering"] = VicSteeringConfig(**kept, **VIC_OPTIONS)
    for name, (path, weight) in VIC_WEIGHTS.items():
        cfg.reward_components[name] = MdpComponent(
            compute_func=vic_cost, dynamic_vars={"value": path}, static_params={"weight": weight})
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    cfg = _s3_agent_config(robot_cfg, env_cfg, args)
    n = robot_cfg.number_of_actions + 2 * len(VicSteeringConfig.vic_joints)
    cfg.model.actor.num_out = n
    cfg.model.actor.mu_model.num_out = n
    # Must match the launcher's ACTOR_LR (fork_checkpoint writes it into the optimizer state).
    cfg.model.actor_optimizer.lr = 5e-5
    return cfg
