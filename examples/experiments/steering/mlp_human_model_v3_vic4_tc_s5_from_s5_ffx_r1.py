"""vic4_TC_s5_from-s5_ffx_r1 (scratch strategy A stage A2 candidate, 2026-11-01; user: "2안 ... 좀 더 공격적으로, 피드포워드
토크가 확확 나와주게 설계 바꿔서 A2 빨리"): vic4_TC_s5_from-s5_noeff_r1 (frozen fixed-PD teacher se-v1-s5, no VIC effort
cost) with the feed-forward torque residual tau_res made able to carry the gait. Reason: in both A2 runs the v4 law at
alpha 1 / beta 0 leaves the legs at human-range stiffness (local s2946 e1145, 1.0 m/s medians: knee 99 -> 70 stance / 16
swing N m/rad, hip flexion 146 -> 113 / 57), and the gait broke exactly in swing (knee 20-30 deg flexed at strike, swing
peak 52 vs 62 deg, foot drop in swing; knee r 0.47); tau_res stayed unused because its outputs were appended with the
parent's log std -2.9 (sigma 0.055) and the actor learned at 2e-5. Changes against noeff (the v4 law itself is unchanged):

- tau_res_scale 0.33 -> 0.5 of the joint strength cap (more feed-forward range per output).
- Feed-forward outputs 69-78 fixed log std -2.9 -> -1.2 (sigma 0.055 -> 0.30, = the co-activation outputs): here through
  PPOActorConfig.logstd_slices, because with a fixed std the agent keeps the config's log std over the checkpoint's (the
  launcher fork sets the same value in the initial checkpoint). Launcher: actor lr 2e-5 -> 5e-5 (the vic_TC_r1-r12 value).
"""
import dataclasses

from examples.experiments.steering import mlp_human_model_v3_vic4_tc_s5_from_s5_noeff_r1 as _noeff
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_s5_from_s5_noeff_r1 import *  # noqa: F401,F403

TAU_RES_SCALE = 0.5
FF_LOGSTD_SLICE = [69, 79, -1.2]


def env_config(robot_cfg, args):
    cfg = _noeff.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(cfg.control_components["steering"], tau_res_scale=TAU_RES_SCALE)
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    cfg = _noeff.agent_config(robot_cfg, env_cfg, args)
    cfg.model.actor.logstd_slices = list(cfg.model.actor.logstd_slices or []) + [FF_LOGSTD_SLICE]
    return cfg
