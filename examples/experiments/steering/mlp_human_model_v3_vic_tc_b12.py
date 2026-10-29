"""vic_TC_b12 (implementation vic_tc_v1 on vic_tc_v1_se, candidate, 2026-10-30): HMV_v1 path B diagnostic run. vic_TC_b4
last.ckpt (cb82cccd..., 80M, the only path-B checkpoint besides b5 40M that keeps quiet standing at all speeds) continued
with only a small actor sensory delay: proprio U(0, 8) ms, vestibular U(0, 6) ms (critic and discriminator true values).
Gains, rewards and commands as b4.

Why: the common update guide asks for the afferent delay in new runs (proprio 40-70 ms, vestibular 30-60 ms), but b8's
weights collapse at a fixed 33/27 ms and lose standing at 16/12 ms, and b9 (b8 + quiet_stance + full delay at once)
fell apart by 10M. b12 is the first rung of a delay ladder from a standing-capable parent (Claude / Codex 2026-10-30);
it is not the final human delay range.
"""
from protomotions.envs.obs.sensory_delay import SensoryDelayConfig, delay_actor_inputs

from examples.experiments.steering import mlp_human_model_v3_vic_tc_b4 as _b4
from examples.experiments.steering.mlp_human_model_v3_vic_tc_b4 import *  # noqa: F401,F403

SENSORY_DELAY = dict(enabled=True, proprio_ms=(0.0, 8.0), vestib_ms=(0.0, 6.0))


def env_config(robot_cfg, args):
    cfg = _b4.env_config(robot_cfg, args)
    cfg.sensory_delay = SensoryDelayConfig(**SENSORY_DELAY)
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    cfg = _b4.agent_config(robot_cfg, env_cfg, args)
    return delay_actor_inputs(cfg, env_cfg.sensory_delay)
