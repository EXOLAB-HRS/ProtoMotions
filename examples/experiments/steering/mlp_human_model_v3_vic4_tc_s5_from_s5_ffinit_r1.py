"""vic4_TC_s5_from-s5_ffinit_r1 (local, 2026-11-01; user-approved A-1 "feed-forward pre-fill"): the noeff A2 recipe started
directly at the final law (alpha 1, beta 0, no ramp) from se-v1-s5 whose 10 feed-forward outputs were pre-fitted offline
(output/261029/e01_vic3_tc_local/configs/ffinit_fit.py) to the torque the v4 gains lack against the fixed PD,
dtau = (k_old - k_act) e - (b_old - b_act) qd, on fixed-PD rollouts of s5 (ridge on the actor's last hidden layer, held-out
1.4 m/s R^2 0.86-0.98). So at alpha 1 / beta 0 the torque k_act e - b_act qd + tau_res starts close to the walker's PD
torque and the policy keeps its gait while it learns. Reasons: ffx showed that exploring feed-forward from 0 with a large
std acts as a disturbance (shuffle at 0.2 m/s); the beta prior is held at the 30 Hz control rate and is not equivalent
to the fixed PD, so the ramp does not preserve the gait. Changes against noeff: tau_res_scale 0.33 -> 1.0 (the fitted
dtau p95 reaches 88 N m at the ankle), ramp off (alpha_init 1, beta_init 0, schedule False); feed-forward log std stays
-2.9 (pre-filled, little exploration needed).
"""
import dataclasses

from examples.experiments.steering import mlp_human_model_v3_vic4_tc_s5_from_s5_noeff_r1 as _noeff
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_s5_from_s5_noeff_r1 import *  # noqa: F401,F403

FINAL_LAW = dict(alpha_init=1.0, beta_init=0.0, schedule=False, tau_res_scale=1.0)


def env_config(robot_cfg, args):
    cfg = _noeff.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(cfg.control_components["steering"], **FINAL_LAW)
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _noeff.agent_config(robot_cfg, env_cfg, args)
