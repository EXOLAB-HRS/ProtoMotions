"""vic2_TC_s5_r7 (vic_tc_v2 candidate, 2026-10-28, server A): the v2 co-activation controller and training pushes only,
with u explored (log std -1.2), no symmetry reward and no upper arms in the discriminator. Why: from vic_TC_s5_r17 epoch 190
(speed 0.081, knee r 0.98, ankle r 0.79) every v2 run lost gait quality: r1 / r4 (pushes, symmetry, arms in disc) speed
0.13, lateral 0.12, ankle r 0.65-0.67; the no-push control r2 kept speed 0.077 and falls 0.16 but its joint curves
collapsed (knee r 0.40, ankle r 0.20). So pushes cost tracking and something in symmetry / arm discriminator costs the leg
curves. r7 keeps only what VIC needs (per-joint u inside the human range, 50/150 ms filter, cost on u, pushes 0.4 m/s) plus
r4's arm gains and r6's u exploration (local r6 tests the same exploration from r15 with symmetry and arms). r8 = r7 +
symmetry, to separate symmetry from the arm discriminator.
"""
from examples.experiments.steering import mlp_human_model_v3_vic2_tc_s5_r1 as _r1
from examples.experiments.steering import mlp_human_model_v3_vic2_tc_s5_r4 as _r4
from examples.experiments.steering.mlp_human_model_v3_vic2_tc_s5_r4 import *  # noqa: F401,F403

_r1.ARMS_IN_DISC = False
_r1.SYM_WEIGHT = 0.0
U_LOGSTD_SLICE = [59, 69, -1.2]   # PPOActorConfig.logstd_slices (ProtoMotions 89813d0), as local r6


def agent_config(robot_cfg, env_cfg, args):
    cfg = _r4.agent_config(robot_cfg, env_cfg, args)
    cfg.model.actor.logstd_slices = [U_LOGSTD_SLICE]
    return cfg
