"""vic3_TC_s5_r10 (vic_tc_v3 candidate, 2026-10-29; user: if the efference copy does not recover what the delay broke,
get a run balanced on W1-W3 fast) = vic3_TC_s5_r4 with the shorter end of the physiological afferent delays:
proprioception 20-40 ms (afferent conduction 15-20 ms + spinal / brainstem relay; was 40-70 ms = transcortical),
vestibular 20-40 ms (was 30-60 ms). Rationale: the single teacher net stands for all CNS loops at once; the later
MC / CB / BS split can give the cortical modules the longer delays and slower rates. Same start (vic3_TC_s5_r3 last);
compare tracking (C) with r4 (40-70 ms) and r11 (no delay).
"""
import dataclasses

from examples.experiments.steering import mlp_human_model_v3_vic3_tc_s5_r4 as _v3r4
from examples.experiments.steering.mlp_human_model_v3_vic3_tc_s5_r4 import *  # noqa: F401,F403

PROPRIO_MS = (20.0, 40.0)
VESTIB_MS = (20.0, 40.0)


def env_config(robot_cfg, args):
    cfg = _v3r4.env_config(robot_cfg, args)
    cfg.sensory_delay = dataclasses.replace(cfg.sensory_delay, proprio_ms=PROPRIO_MS, vestib_ms=VESTIB_MS)
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _v3r4.agent_config(robot_cfg, env_cfg, args)
