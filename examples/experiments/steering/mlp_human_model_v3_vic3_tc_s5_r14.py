"""vic3_TC_s5_r14 (vic_tc_v3 candidate, 2026-10-30, server; user approved: short delay + efference copy, long training
continued from a good run) = vic3_TC_s5_r10 delays (proprioception / vestibular 20-40 ms) + vic3_TC_s5_r7 full efference
copy (last 3 PD actions 59 x 3 and u / tau_ff 20 x 3). r10 gave the best teacher_score_v2 so far (51; speed / lateral
0.116 / 0.087, falls 0.126/min, W2 pass); the efference copy kept r7's gait (knee r 0.965) but never fixed tracking alone.
No new mechanism. Continued from vic3_TC_s5_r10 last for 100M: in every vic3 run the last checkpoint was best and W1-W3
evals still improved by epoch. r10's checkpoint has no efference columns: the launcher appends 177 + 60 zero-weight ones.
vic3_TC_s5_r15 = the same recipe with a second seed (2945). Pulse error cost (r12) and u / damping changes are left out
until decided.
"""
import dataclasses

from examples.experiments.steering import mlp_human_model_v3_vic3_tc_s5_r7 as _v3r7
from examples.experiments.steering import mlp_human_model_v3_vic3_tc_s5_r10 as _v3r10
from examples.experiments.steering.mlp_human_model_v3_vic3_tc_s5_r7 import *  # noqa: F401,F403


def env_config(robot_cfg, args):
    cfg = _v3r7.env_config(robot_cfg, args)
    cfg.sensory_delay = dataclasses.replace(cfg.sensory_delay, proprio_ms=_v3r10.PROPRIO_MS, vestib_ms=_v3r10.VESTIB_MS)
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _v3r7.agent_config(robot_cfg, env_cfg, args)
