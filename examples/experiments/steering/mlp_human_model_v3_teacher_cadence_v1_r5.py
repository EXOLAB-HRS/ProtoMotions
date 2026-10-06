"""teacher_cadence_v1 revision 5: r4 (r3 recipe + random actor cadence columns) with actor lr 1e-4.

r4 (261003 e02) learned no rho* response; in 20M steps its actor's first layer moved 4.3%
(critic 12%) at actor lr 2e-5. r5 raises the actor lr to the critic's 1e-4; the forked
checkpoint's optimizer groups get the same value (teacher_followup.py ACTOR_LR).
"""
from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r3 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r3 import (
    env_config, agent_config as _r3_agent_config)

ACTOR_LR = 1e-4


def agent_config(robot_cfg, env_cfg, args):
    cfg = _r3_agent_config(robot_cfg, env_cfg, args)
    cfg.model.actor_optimizer.lr = ACTOR_LR
    return cfg
