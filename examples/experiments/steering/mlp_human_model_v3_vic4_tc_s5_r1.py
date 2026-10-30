"""vic4_TC_s5_r1 (implementation vic_tc_v4, user-approved version 2026-10-30: impedance prior) = vic2_TC_s5_r11 with the
v4 stiffness / damping law (vic_steering.py, vic_law = "v4"), no afferent delay (user 2026-10-30: the teacher may stay
undelayed; MC / CB add the delay with a predictor later), 100M from vic2_TC_s5_r11 last (384be4d4...).

    ln k_tot = ln k_c(phi) + (h(phi) + ln 2) a,   k_tot = plant passive (human_model_v3 tissue tangent) + VIC PD
    ln b_tot = ln b_c(phi) + alpha (ln k_tot - ln k_c(phi))   (D1; ankle alpha 0.15, knee / hip 0.5)
    r_imp = exp(-mean_j (dead-zone log distance outside the HMV-ImpRef range / 0.5)^2), weight IMP_WEIGHT

Why: v1 / v2 made k a lookup of the human reference by discrete leg state (W3 band passes by construction, policy delta
~ 0), v3's u never rose (pulses, activation cost, softplus, pulse-error cost: 4 runs, vic3 r4-r12). v4 lets the policy
move k inside twice the human range and a dead-zone reward (zero inside the range, like a motion prior) keeps the total
joint stiffness human-like; the phase weights are continuous (load fraction from the GRF, no state switch), the plant's
passive stiffness is no longer counted twice, and co-contraction (tau_co beyond the load stiffness c |tau|) is an output
for J2M instead of a separate control. Parent r11: best undelayed v2 run under teacher_score_v2 (70; H 75 C 62 S 57 I 88,
E 81). Its a ~ 0 maps to the reference centre, so the first actions keep the parent's stiffness (minus the plant's passive
part, which was double-counted). The band cost (same dead zone, linear) is replaced by r_imp. Validation of k / b is on
data the reward does not use: impedance_probe_v1 E1 standing / walking, other speeds, W4 exo response.
Plant: human_model_v3.2 (v3.1 + the joint spinal stretch reflex of vic2_TC_s5_r14b, training-calibrated thresholds; silent
in steady walking), as in the vic3 runs; the reflex gain uses the v4 co-contraction level u = tau_co / tau_max.
"""
import dataclasses

from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.control.vic_steering import vic_gain, VIC_IMP_PRIOR
from protomotions.robot_configs.human_model.common.joint_reflex import JointReflexConfig

from examples.experiments.steering import mlp_human_model_v3_vic2_tc_s5_r11 as _r11
from examples.experiments.steering import mlp_human_model_v3_vic2_tc_s5_r14b as _r14b
from examples.experiments.steering.mlp_human_model_v3_vic2_tc_s5_r11 import *  # noqa: F401,F403

IMP_WEIGHT = 0.1
VIC4 = dict(vic_law="v4", v4_band_margin=2.0, v4_load_lo=0.05, v4_load_hi=0.20, v4_load_tau_s=0.03,
            v4_late_anchor=True, v4_late_p=(0.5, 0.65), v4_stance_T0_s=0.65, v4_k_floor_frac=0.02, v4_imp_scale=0.5,
            v4_passive=True)


def env_config(robot_cfg, args):
    cfg = _r11.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(cfg.control_components["steering"], **VIC4)
    cfg.reward_components.pop("vic_band_cost", None)
    cfg.reward_components["vic_imp_prior"] = MdpComponent(
        compute_func=vic_gain, dynamic_vars={"value": VIC_IMP_PRIOR}, static_params={"weight": IMP_WEIGHT})
    cfg.joint_reflex = JointReflexConfig(**_r14b.JOINT_REFLEX)
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _r11.agent_config(robot_cfg, env_cfg, args)
