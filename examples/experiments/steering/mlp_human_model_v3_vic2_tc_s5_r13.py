"""vic2_TC_s5_r13 (vic_tc_v2 candidate, draft 2026-10-28, user-approved sensory delay): vic2_TC_s5_r11 (no training
pushes) with a physiological afferent delay on the actor's body-state observations (protomotions/envs/obs/sensory_delay.py).

Why: in 15+ v2 runs u (co-activation level) never rose after a push (coact_push_rise <= 0.002). The policy sees the
body state at once and can correct through its PD targets every 33 ms, so pre-stiffening never pays. A person's
afferent feedback arrives late (transcortical loop, afferent + central part ~40-70 ms proprioceptive, ~30-60 ms
vestibular), so when an error shows up the correction comes too late and a stiffer joint is the cheaper answer.
r13 gives the actor that delay: proprio (max_coords_obs, historical_max_coords_obs) d ~ U(40, 70) ms and vestibular
(se_gravity, se_specific_force) d ~ U(30, 60) ms, drawn per env at reset, linear interpolation between 30 Hz slots.
Command (steering, cadence_cond), gait clock (cadence_phase) and terrain are not delayed. The critic, the AMP
discriminator and its critic keep the undelayed observations. The efferent side is unchanged (u filter 50 / 150 ms,
tau_ff 30 ms). Evaluate with fixed_proprio_ms / fixed_vestib_ms (e.g. 55 / 45) in env.sensory_delay.
The launcher (teacher_followup.py) must list "vic2-tc-s5-r13" with r11's entries (ACTOR_LR, U_LOGSTD, VIC2_LEGS_DISC).
"""
from protomotions.envs.obs.sensory_delay import SensoryDelayConfig, delay_actor_inputs

from examples.experiments.steering import mlp_human_model_v3_vic2_tc_s5_r11 as _r11
from examples.experiments.steering.mlp_human_model_v3_vic2_tc_s5_r11 import *  # noqa: F401,F403

SENSORY_DELAY = dict(enabled=True, proprio_ms=(40.0, 70.0), vestib_ms=(30.0, 60.0))


def env_config(robot_cfg, args):
    cfg = _r11.env_config(robot_cfg, args)
    cfg.sensory_delay = SensoryDelayConfig(**SENSORY_DELAY)
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    cfg = _r11.agent_config(robot_cfg, env_cfg, args)
    return delay_actor_inputs(cfg, env_cfg.sensory_delay)
