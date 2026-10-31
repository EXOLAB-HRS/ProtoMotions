"""vic4_TC_walkA1_from-walkr2_r1 (scratch strategy A, stage A2, candidate 2026-10-31; user 2026-10-30 "걷기 먼저 → 고정 →
VIC", 2026-10-31 "A1b 수렴하면 평가 빨리 하고 VIC 있는 A2로"): VIC v4 trained fresh from the frozen fixed-PD walker
walk_TC_scratch_from-walkr1_r2 (A1b, knee_strike_prior 0.6). Only the controller changes against the walker:

- Stiffness / damping: the vic_tc_v4 law of vic4_TC_s5_r1 / r4 unchanged (phi continuous load blend, ankle late anchor,
  D1 damping, passive part counted once, 50 / 150 ms activation filter, impedance prior 0.1, co-contraction cost 0.03,
  VIC effort cost 0.05). The band cost is really removed (its key is "vic_band"; vic4_TC_s5_r1 popped "vic_band_cost").
- Start from fixed PD with the existing vic_tc_v1 schedule (as vic_TC_s5_r15 from se-v1-s5): alpha 0 -> 1 (gains move in
  log space from the walker's pc_240_4_r3 PD to the v4 law, 9000 control steps), then beta 1 -> 0 (torque prior that
  keeps the walker's PD torque, 9000 steps), both paused while the fall rate rises. The launcher appends the 20 VIC
  actor outputs zero-initialised (a = 0, no feed-forward residual), co-activation log std -1.2.
- Plant human_model_v3.2 = v3.1 + the joint spinal stretch reflex of vic2_TC_s5_r14b (as every vic3 / vic4 run; the
  walker docstring defers the reflex to this stage because with fixed PD it only acts through the VIC feed-forward path).

Kept from the walker (not from the vic2 / vic4 chain): its command set and cadence law (pack law f0 = 0.608 v + 1.178,
not r17's 0.558 v + 1.060), its reward weights, legs discriminator 2 steps, and the pc_240_4_r3 arm gains (the vic2_TC_s5_r4
arm-gain plant change is not applied, so the walker's plant stays except for the VIC joints). No sensory delay, no pushes,
no symmetry reward. New-run defaults: AMP replay fix and desynced first timeouts on.
"""
import dataclasses

from examples.experiments.steering import mlp_human_model_v3_vic2_tc_s5_r1 as _v2r1
from examples.experiments.steering import mlp_human_model_v3_vic4_tc_s5_r1 as _v4r1
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_s5_r1 import *  # noqa: F401,F403

# The walker's (teacher_se_v1 s5) cadence law, overriding r17's human law inherited through the vic chain.
WALKER_CADENCE = dict(cadence_f0_slope=0.608, cadence_f0_intercept=1.178)
# vic_tc_v1 ramp from fixed PD (VicSteeringConfig defaults; vic_TC_s5_r15), instead of r17's state-file alpha 1 / beta 0.
RAMP = dict(alpha_init=0.0, beta_init=1.0, schedule=True, alpha_steps=9000, beta_steps=9000)
KNEE_STRIKE_W = 0.6          # = the parent walker's weight
PLANT_GAIN_OVERRIDES = {}    # walker arm gains (pc_240_4_r3); the launcher reads this module attribute


def configure_robot_and_simulator(robot_cfg, simulator_cfg, args):
    _v2r1.configure_robot_and_simulator(robot_cfg, simulator_cfg, args)   # stage PD / plant setup, pushes off (r11)


def env_config(robot_cfg, args):
    cfg = _v4r1.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(cfg.control_components["steering"], **WALKER_CADENCE, **RAMP)
    cfg.reward_components.pop("vic_band", None)
    cfg.reward_components["knee_strike_prior"].static_params["weight"] = KNEE_STRIKE_W
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _v4r1.agent_config(robot_cfg, env_cfg, args)
