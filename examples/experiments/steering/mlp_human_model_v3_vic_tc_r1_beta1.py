"""vic_TC_r1 comparison arm (candidate, 2026-10-22): same as vic_tc_r1 with beta held at 1.

alpha still moves the gains to the human reference, but the r25-equivalent feed-forward torque is never
removed. Separates the effect of soft drive gains alone from the policy taking the torque over.
"""
from examples.experiments.steering import mlp_human_model_v3_vic_tc_r1 as _vic
from examples.experiments.steering.mlp_human_model_v3_vic_tc_r1 import *  # noqa: F401,F403

_vic.BETA_DECAY = False
env_config = _vic.env_config
agent_config = _vic.agent_config
