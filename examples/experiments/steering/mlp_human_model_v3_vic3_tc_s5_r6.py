"""vic3_TC_s5_r6 (vic_tc_v3 candidate, 2026-10-29, local PC) = vic3_TC_s5_r3 without the joint spinal reflex:
vic3_TC_s5_r2 (stance load term) + motor noise 5 % + standing sway cost 0.05, from vic2_TC_s5_r11 last like r3.
Question: r3 is the first run to pass W2 (knee r 0.958, ankle r 0.905; r2 0.563 / 0.524). Which of its two additions,
the reflex (human_model_v3.2) or the reasons to co-contract (noise, sway), made the leg curves human-like?
Same seed, envs, length and parent as r3; compare with r3 and r2.
"""
import dataclasses

from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.control.vic_steering import vic_cost, VIC_SWAY

from examples.experiments.steering import mlp_human_model_v3_vic3_tc_s5_r2 as _v3r2
from examples.experiments.steering import mlp_human_model_v3_vic3_tc_s5_r3 as _v3r3
from examples.experiments.steering.mlp_human_model_v3_vic3_tc_s5_r2 import *  # noqa: F401,F403


def env_config(robot_cfg, args):
    cfg = _v3r2.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(
        cfg.control_components["steering"], v3_motor_noise=_v3r3.MOTOR_NOISE, v3_motor_noise_tau_s=0.04,
        sway_cost=True, sway_v_h=0.01)
    cfg.reward_components["vic_sway"] = MdpComponent(
        compute_func=vic_cost, dynamic_vars={"value": VIC_SWAY}, static_params={"weight": _v3r3.SWAY_WEIGHT})
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _v3r2.agent_config(robot_cfg, env_cfg, args)
