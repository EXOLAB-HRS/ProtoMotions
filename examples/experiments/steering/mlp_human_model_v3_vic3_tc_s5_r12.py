"""vic3_TC_s5_r12 (vic_tc_v3 candidate, 2026-10-30, server) = vic3_TC_s5_r7 + the kinematic-error cost under the pulses
of vic3_TC_s5_r8 (pulse_err_cost, weight 0.05). r7 (full efference copy) gave the best gait curves (knee r 0.965,
ankle r 0.921, teacher_score_v2 46) but more falls (0.32/min); r8 (pulse error cost) gave the fewest falls of the pulse
runs (0.166/min). Both are tested candidates; this run asks whether they add up. Same start (vic3_TC_s5_r3 last), the
launcher appends r7's 177 + 60 zero-weight actor input columns. Named r12 because the local PC took r11 (r7 continuation).
"""
import dataclasses

from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.control.vic_steering import vic_cost, VIC_PULSE_ERR

from examples.experiments.steering import mlp_human_model_v3_vic3_tc_s5_r7 as _v3r7
from examples.experiments.steering import mlp_human_model_v3_vic3_tc_s5_r8 as _v3r8
from examples.experiments.steering.mlp_human_model_v3_vic3_tc_s5_r7 import *  # noqa: F401,F403


def env_config(robot_cfg, args):
    cfg = _v3r7.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(cfg.control_components["steering"], pulse_err_cost=True)
    cfg.reward_components["vic_pulse_err"] = MdpComponent(
        compute_func=vic_cost, dynamic_vars={"value": VIC_PULSE_ERR}, static_params={"weight": _v3r8.PULSE_ERR_WEIGHT})
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _v3r7.agent_config(robot_cfg, env_cfg, args)
