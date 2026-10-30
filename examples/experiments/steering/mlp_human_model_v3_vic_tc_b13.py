"""vic_TC_b13 (implementation vic_tc_v1 on vic_tc_v1_se, candidate, 2026-10-30): HMV_v1 path B, delay ladder rung 2.
vic_TC_b12 last.ckpt (d0b6e38a..., b4 + 15M with proprio U(0, 8) / vestibular U(0, 6) ms) continued with the actor
sensory delay widened to proprio U(0, 16) ms, vestibular U(0, 12) ms. Gains, rewards and commands as b4 / b12.

Why: b12 kept quiet standing at the training delay (15M: 35-42 of 64) and, at a fixed 16/12 ms, still had 29-31 of 64
with 3 of 64 envs falling (start rule passed). The range keeps the previous 0-8 / 0-6 ms and only raises the upper end
(Claude / Codex 2026-10-30). The target human range (40-70 / 30-60 ms) is still ahead.
"""
from examples.experiments.steering import mlp_human_model_v3_vic_tc_b12 as _b12
from examples.experiments.steering.mlp_human_model_v3_vic_tc_b12 import *  # noqa: F401,F403

SENSORY_DELAY = dict(enabled=True, proprio_ms=(0.0, 16.0), vestib_ms=(0.0, 12.0))


def env_config(robot_cfg, args):
    cfg = _b12._b4.env_config(robot_cfg, args)
    cfg.sensory_delay = SensoryDelayConfig(**SENSORY_DELAY)
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    cfg = _b12._b4.agent_config(robot_cfg, env_cfg, args)
    return delay_actor_inputs(cfg, env_cfg.sensory_delay)
