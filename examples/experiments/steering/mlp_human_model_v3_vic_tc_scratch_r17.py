"""vic_TC_scratch_r17 (implementation vic_tc_v1 on vic_tc_v1_se, candidate, 2026-10-27, hyojun): strategy C stage 2.
Continues vic_TC_scratch_r16 last.ckpt with teacher_se_v1_s3 rewards and commands (r25's full command set, stops,
turns, cadence 0.3 m/s lower bound, steady speed term), as se-v1 s1 -> s2/s3. VIC unchanged from r16 (alpha = 1,
beta = 0, delta_band, 15/50 ms delta filter, frontal human reference, damping law 0.5), actor lr 2e-5.
"""
from examples.experiments.steering.mlp_human_model_v3_teacher_se_v1_s3 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_teacher_se_v1_s3 import (
    env_config as _s3_env_config, agent_config as _s3_agent_config)
from examples.experiments.steering.mlp_human_model_v3_vic_tc_scratch_r16 import (  # noqa: F401
    add_vic, widen_actor, VIC_OPTIONS, VIC_WEIGHTS, ACTOR_LR)


def env_config(robot_cfg, args):
    return add_vic(_s3_env_config(robot_cfg, args))


def agent_config(robot_cfg, env_cfg, args):
    return widen_actor(_s3_agent_config(robot_cfg, env_cfg, args), robot_cfg)
