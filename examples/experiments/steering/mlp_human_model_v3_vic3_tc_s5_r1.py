"""vic3_TC_s5_r1 (implementation vic_tc_v3, candidate DRAFT 2026-10-28; v3 structure approved by the user 2026-10-28).

= vic2_TC_s5_r11 (no training pushes, r4 arm gains, u log std -1.2, legs-only discriminator)
  + the afferent delay of vic2_TC_s5_r13 (proprio U(40, 70) ms, vestibular U(30, 60) ms on the actor only)
  + the v3 stiffness law (vic_steering.py, vic_law = "v3"):
        k_p = k_base + c |tau|_f + u k_co_max,   k_d = d sqrt(k_p),   hard corridor [0.3 lo_s, 2 hi_s]
    with the band cost replaced by a dead-zone guard (zero inside [lo / 1.25, 1.25 hi] of HMV-ImpRef, never toward the
    centre) and the cocontraction cost on the co-contraction torque share k_co / (c tau_max)
  + the left-right symmetry reward back at weight 0.2 with the human dead zone (gait_symmetry.py: per-joint SYM_TOL_DEG,
    stance % factor with tol 3.73 %).

Why: in v2 k is a lookup of the human reference by leg state (W3 passes by construction), the reference values include
reflex contributions, and u never moved (coact_push_rise <= 0.002 in 15+ runs). In v3 the phase variation of k comes
from the joint's own load (short-range stiffness ~ muscle force) and the policy's u only adds co-contraction, which
costs; standing ankle stiffness (HMV-ImpRef 290-340 N m/rad at ~8 N m) needs u ~ 0.6, so u has a job. The afferent delay
makes pre-stiffening pay (r13 rationale). The parent's u outputs (tanh(a) ~ -0.1 .. -0.7) map to u = 0 (relu), so the run
starts relaxed + load term.

Not yet included: the joint spinal-reflex block (another worktree; flag added later). The launcher (teacher_followup.py)
must list "vic3-tc-s5-r1" with r11's entries (ACTOR_LR, U_LOGSTD, VIC2_LEGS_DISC) and r13's delay settings.
Evaluate the gain log keys k_base / k_load / k_co / tau_load to show where the phase variation comes from.
"""
import dataclasses

from examples.experiments.steering import mlp_human_model_v3_vic2_tc_s5_r1 as _r1
from examples.experiments.steering import mlp_human_model_v3_vic2_tc_s5_r13 as _r13
from examples.experiments.steering.mlp_human_model_v3_vic2_tc_s5_r13 import *  # noqa: F401,F403

_r1.SYM_WEIGHT = 0.2      # symmetry reward back on (r7 / r11 had 0), now with the human dead zone

VIC3 = dict(vic_law="v3", v3_hard_lo=0.3, v3_hard_hi=2.0, v3_guard_margin=1.25,
            v3_load_tau_up_s=0.05, v3_load_tau_down_s=0.15, v3_load_source="sim", v3_srs_omega=0.0,
            v3_u_map="relu", v3_coact_cost="torque")
SYM = dict(sym_deadzone=True, sym_stance_tol_pct=3.73, sym_stance_sigma_pct=2.0)


def env_config(robot_cfg, args):
    cfg = _r13.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(cfg.control_components["steering"], **VIC3, **SYM)
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _r13.agent_config(robot_cfg, env_cfg, args)
