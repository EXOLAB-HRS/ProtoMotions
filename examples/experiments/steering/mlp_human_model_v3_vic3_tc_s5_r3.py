"""vic3_TC_s5_r3 (vic_tc_v3 candidate, 2026-10-29; user 2026-10-29: u should rise where a person co-contracts, and the
spinal reflex belongs in the v3 line) = vic3_TC_s5_r2 (stance-only load term)
  + the joint spinal stretch reflex of vic2_TC_s5_r14b (human_model_v3.2, high-pass signal, training-calibrated v_th)
  + reasons to co-contract, without training pushes:
      signal-dependent motor noise 5 % of (|tau| + tau_co), 40 ms colour (v3_motor_noise): stiffness reduces its effect,
        and with the 40-70 ms afferent delay the policy cannot cancel it by q_ref in time (Selen et al. 2009);
        5 % = upper end of human force variability at moderate force (Jones et al. 2002).
      quiet-standing sway cost (sway_cost, weight 0.05, dead zone 10 mm/s): a person keeps sway small when standing;
        standing ankle stiffness (HMV-ImpRef 290-340 N m/rad) needs u ~ 0.6 under the v3 law.
Expected where u should rise: standing ankle (sway + noise), loading response (delay + noise at contact).
Checks: criteria rows u_stand_ankle, knee_u_loading_over_swing, k_push_rise (check_criteria.py, 2026-10-29).
"""
import dataclasses

from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.control.vic_steering import vic_cost, VIC_SWAY
from protomotions.robot_configs.human_model.common.joint_reflex import JointReflexConfig

from examples.experiments.steering import mlp_human_model_v3_vic3_tc_s5_r2 as _v3r2
from examples.experiments.steering import mlp_human_model_v3_vic2_tc_s5_r14b as _r14b
from examples.experiments.steering.mlp_human_model_v3_vic3_tc_s5_r2 import *  # noqa: F401,F403

MOTOR_NOISE = 0.05
SWAY_WEIGHT = 0.05


def env_config(robot_cfg, args):
    cfg = _v3r2.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(
        cfg.control_components["steering"], v3_motor_noise=MOTOR_NOISE, v3_motor_noise_tau_s=0.04, sway_cost=True,
        sway_v_h=0.01)
    cfg.reward_components["vic_sway"] = MdpComponent(
        compute_func=vic_cost, dynamic_vars={"value": VIC_SWAY}, static_params={"weight": SWAY_WEIGHT})
    cfg.joint_reflex = JointReflexConfig(**_r14b.JOINT_REFLEX)
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _v3r2.agent_config(robot_cfg, env_cfg, args)
