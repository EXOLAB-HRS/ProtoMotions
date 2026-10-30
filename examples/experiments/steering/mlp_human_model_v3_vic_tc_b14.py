"""vic_TC_b14 (implementation vic_tc_v1 on vic_tc_v1_se, candidate, 2026-10-30): HMV_v1 path B control for vic_TC_b13.
vic_TC_b12 last.ckpt (d0b6e38a...) continued with b13's delay (proprio U(0, 16) ms, vestibular U(0, 12) ms) and only the
actor learning rate lowered 2e-5 -> 1e-5. Epoch checkpoints every 2.5M transitions (19 epochs at 4096 envs) instead of 5M,
to see the standing swings at a finer time step. Everything else as b13.

Why: b12 and b13 both went standing pass at 5M -> fail at 10M; b13 15M (judged) failed with standing ref cycle 0.41-0.44
and W2 fails 13|11|7. Codex 2026-10-30: training metrics swing with a ~9.3-epoch period (299-step episode / 32-step
rollouts, synchronised resets), actor clip fraction does not jump, lr was 2e-5 in both. b14 asks whether a lower actor lr
alone damps the swing. Reset policy and the AMP replay path are left unchanged (one factor).
"""
from examples.experiments.steering import mlp_human_model_v3_vic_tc_b13 as _b13
from examples.experiments.steering.mlp_human_model_v3_vic_tc_b13 import *  # noqa: F401,F403

ACTOR_LR = 1e-5


def agent_config(robot_cfg, env_cfg, args):
    cfg = _b13.agent_config(robot_cfg, env_cfg, args)
    # Must match the launcher's ACTOR_LR (fork_checkpoint writes it into the optimizer state).
    cfg.model.actor_optimizer.lr = ACTOR_LR
    cfg.save_epoch_checkpoint_every = max(1, round(2_500_000 / (args.num_envs * 32)))
    return cfg
