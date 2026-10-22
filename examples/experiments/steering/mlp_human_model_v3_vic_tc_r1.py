"""vic_TC_r1 (implementation vic_tc_v1, candidate, 2026-10-22): teacher_cadence_v1 r25 with variable joint impedance.

Same rewards, commands, motion pack and AMP as r25. The policy gains 20 outputs for 10 joints (both legs Hip_y,
Knee_y, Ankle_y with human impedance references; Hip_x, Subtalar_x with the current gain as reference): a stiffness
residual and a feed-forward torque residual (protomotions/envs/control/vic_steering.py). alpha moves the gain
reference from the current gains to the human reference, then beta removes the r25-equivalent feed-forward
torque so the policy produces it. Three small penalties: gains outside the tolerated band, stiffening above the
reference (co-contraction cost) and squared active torque on the 10 joints (effort).
Starts from r25 last.ckpt with the new actor outputs at zero (identical actions at alpha = 0).
"""
import dataclasses

from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r25 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r25 import (
    env_config as _parent_env_config, agent_config as _parent_agent_config)
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.control.vic_steering import (
    VicSteeringConfig, vic_cost, VIC_BAND_COST, VIC_COCONTRACTION, VIC_EFFORT)

BETA_DECAY = True
VIC_WEIGHTS = {"vic_band": (VIC_BAND_COST, 0.03), "vic_cocontraction": (VIC_COCONTRACTION, 0.03),
               "vic_effort": (VIC_EFFORT, 0.05)}


def env_config(robot_cfg, args):
    cfg = _parent_env_config(robot_cfg, args)
    old = cfg.control_components["steering"]
    kept = {f.name: getattr(old, f.name) for f in dataclasses.fields(old) if f.name != "_target_"}
    cfg.control_components["steering"] = VicSteeringConfig(**kept, beta_decay=BETA_DECAY)
    for name, (path, weight) in VIC_WEIGHTS.items():
        cfg.reward_components[name] = MdpComponent(
            compute_func=vic_cost, dynamic_vars={"value": path}, static_params={"weight": weight})
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    cfg = _parent_agent_config(robot_cfg, env_cfg, args)
    n = robot_cfg.number_of_actions + 2 * len(VicSteeringConfig.vic_joints)
    cfg.model.actor.num_out = n
    cfg.model.actor.mu_model.num_out = n
    # New outputs must learn while alpha/beta move; r25's 2e-5 moved the actor ~4 % in 20M (r4).
    # Must match the launcher's ACTOR_LR (fork_checkpoint writes it into the optimizer state).
    cfg.model.actor_optimizer.lr = 5e-5
    return cfg
