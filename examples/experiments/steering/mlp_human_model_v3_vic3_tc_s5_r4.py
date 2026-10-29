"""vic3_TC_s5_r4 (vic_tc_v3 candidate, 2026-10-29; user: design so that u actually rises, run now) = vic3_TC_s5_r3
(stance load term, joint reflex, motor noise 5 %, standing sway cost; first W2 pass) + three changes against the
three reasons u stayed flat in r1-r3 and every v2 run:
  (a) no reason in time: unpredictable joint torque pulses shorter than afferent delay + one control step
      (v3_perturb_*: 1 Hz per joint, 8 % of the effort limit x U(0.5, 1), 60 ms) only while standing and in the first
      0.15 s of stance; q_ref cannot cancel them in time, stiffness can (Burdet et al. 2001, Franklin et al. 2003).
      Joint level and small, not the BS push.
  (b) u dearer than tau_ff: co-contraction priced as its extra activation, ((|tau| + tau_co)^2 - tau^2) / tau_max^2,
      at the vic_effort weight 0.05 (v3_coact_cost="activation"), so the same torque costs the same either way.
  (c) flat zero region: u = softplus(8 tanh(a)) / 8 (v3_u_map="softplus").
Started from vic3_TC_s5_r3 last. Checks: u_stand_ankle, knee_u_loading_over_swing, k_push_rise; standing sway.
"""
import dataclasses

from examples.experiments.steering import mlp_human_model_v3_vic3_tc_s5_r3 as _v3r3
from examples.experiments.steering.mlp_human_model_v3_vic3_tc_s5_r3 import *  # noqa: F401,F403

PERTURB = dict(v3_perturb_rate_hz=1.0, v3_perturb_amp=0.08, v3_perturb_dur_s=0.06, v3_perturb_loading_s=0.15)
COCONTRACTION_WEIGHT = 0.05   # = vic_effort weight


def env_config(robot_cfg, args):
    cfg = _v3r3.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(
        cfg.control_components["steering"], v3_coact_cost="activation", v3_u_map="softplus", **PERTURB)
    cfg.reward_components["vic_cocontraction"].static_params["weight"] = COCONTRACTION_WEIGHT
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _v3r3.agent_config(robot_cfg, env_cfg, args)
