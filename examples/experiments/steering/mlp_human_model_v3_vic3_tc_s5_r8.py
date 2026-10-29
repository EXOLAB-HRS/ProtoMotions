"""vic3_TC_s5_r8 (vic_tc_v3 candidate, 2026-10-29) = vic3_TC_s5_r4 + kinematic-error cost under the pulses
(pulse_err_cost, weight 0.05 like vic_effort): while a pulse is live on a joint, ((qdot - qdot at onset) / 1 rad/s)^2.
Why: in r4 the early-stance pulses cost only through falls and the gait prior (the sway cost acts only in standing),
so a stiffer joint at contact barely paid. People scale co-contraction with the kinematic error an unpredictable load
causes (Franklin et al. 2008). Same start (vic3_TC_s5_r3 last); compare with r4.
"""
import dataclasses

from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.control.vic_steering import vic_cost, VIC_PULSE_ERR

from examples.experiments.steering import mlp_human_model_v3_vic3_tc_s5_r4 as _v3r4
from examples.experiments.steering.mlp_human_model_v3_vic3_tc_s5_r4 import *  # noqa: F401,F403

PULSE_ERR_WEIGHT = 0.05


def env_config(robot_cfg, args):
    cfg = _v3r4.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(cfg.control_components["steering"], pulse_err_cost=True)
    cfg.reward_components["vic_pulse_err"] = MdpComponent(
        compute_func=vic_cost, dynamic_vars={"value": VIC_PULSE_ERR}, static_params={"weight": PULSE_ERR_WEIGHT})
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _v3r4.agent_config(robot_cfg, env_cfg, args)
