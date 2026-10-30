"""vic4c_TC_scratch_from-vic4cr1_r2 (scratch strategy A, stage A1b, candidate 2026-10-30; user: a W1 / W2 teacher from
scratch first, then extend to VIC v4, no afferent delay). Continues A1a (vic4c_TC_scratch_r1 last) with the full command
set and rewards of vic4_TC_s5_r1 (= teacher_se_v1 s5 rewards, legs discriminator 2 steps, human_model_v3.2 joint reflex).

Stiffness / damping: the vic_tc_v4 law with the policy's 10 stiffness outputs ignored (v4_freeze_a, a = 0), so
k_tot = k_c(phi) and b_tot = b_c(phi), the HMV-ImpRef centre blended by the GRF load fraction (plant passive part
subtracted). The gait is learned on the impedance it keeps in A2, instead of fixed PD gains (hip 146, knee 99, ankle 79
N m/rad) that every VIC fine-tune so far had to leave (vic3_TC_s5_r1: knee r 0.98 -> -0.20). The feed-forward torque
residual (10 outputs) is trained from the start, as in A2. Stage A2 (vic4_TC_scratch_from-vic4cr2_r3) unfreezes a with
the stiffness rows zeroed at the fork, so A2 starts exactly at this gait.
Also: the band cost key is "vic_band" (vic4_TC_s5_r1 popped "vic_band_cost", so its band cost 0.03 stayed on; here it is
removed as that recipe intended; r_imp stays, constant 1 while a = 0). VIC schedule off (alpha 1, beta 0 from the start).
"""
import dataclasses

from examples.experiments.steering import mlp_human_model_v3_vic4_tc_s5_r1 as _v4r1
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_s5_r1 import *  # noqa: F401,F403

FREEZE = dict(v4_freeze_a=True, schedule=False, alpha_init=1.0, beta_init=0.0)


def env_config(robot_cfg, args):
    cfg = _v4r1.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(cfg.control_components["steering"], **FREEZE)
    cfg.reward_components.pop("vic_band", None)
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _v4r1.agent_config(robot_cfg, env_cfg, args)
