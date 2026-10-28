"""vic3_TC_s5_r2 (implementation vic_tc_v3, candidate 2026-10-29) = vic3_TC_s5_r1 + v3_load_gate = "stance".

One change: the load term c |tau|_f gets torque input only while the leg is in stance or standing (vic_steering.py
v3_load_gate). r1 (local): the swing knee load term was 13.6 N m/rad (human swing 0.5-3.5), k sat on the corridor ceiling
98 % of steps, swing hip 41 %; knee / ankle curves collapsed after epoch 304 (r -0.20 / 0.11 at last). In swing the
active torque is mostly the PD's own k * err, so the load term fed back on itself; human short-range stiffness follows
muscle force, which in swing is small. Stance-late ankle rise from load (r1: 175 N m/rad) is kept.
Server run: 4096 env / 16384 batch (r1 local ran 2048 / 8192; compare with that noted).
"""
import dataclasses

from examples.experiments.steering import mlp_human_model_v3_vic3_tc_s5_r1 as _v3r1
from examples.experiments.steering.mlp_human_model_v3_vic3_tc_s5_r1 import *  # noqa: F401,F403


def env_config(robot_cfg, args):
    cfg = _v3r1.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(cfg.control_components["steering"], v3_load_gate="stance")
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _v3r1.agent_config(robot_cfg, env_cfg, args)
