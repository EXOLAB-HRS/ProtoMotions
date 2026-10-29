"""vic3_TC_s5_r7 (vic_tc_v3 candidate, 2026-10-29; user: the efference copy should include u and tau_ff) = vic3_TC_s5_r5
+ the efference copy of the VIC outputs: the u command and tau_ff target / tau_max of the last 3 steps (20 x 3 = 60,
undelayed, VicSteering.vic_efference_steps). With r5's 59 x 3 PD actions the actor now sees its whole last motor
command (79 x 3). Same start (vic3_TC_s5_r3 last); the launcher appends 177 + 60 zero-weight actor input columns.
"""
import dataclasses

from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.control.vic_steering import compute_vic_efference_obs, VIC_EFFERENCE

from examples.experiments.steering import mlp_human_model_v3_vic3_tc_s5_r5 as _v3r5
from examples.experiments.steering.mlp_human_model_v3_vic3_tc_s5_r5 import *  # noqa: F401,F403

VIC_EFFERENCE_KEY = "vic_efference_copy"


def env_config(robot_cfg, args):
    cfg = _v3r5.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(cfg.control_components["steering"], vic_efference_steps=3)
    cfg.observation_components[VIC_EFFERENCE_KEY] = MdpComponent(
        compute_func=compute_vic_efference_obs, dynamic_vars={"vic_efference": VIC_EFFERENCE})
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    cfg = _v3r5.agent_config(robot_cfg, env_cfg, args)
    m = cfg.model
    for module in (m.actor, m.actor.mu_model):
        module.in_keys = list(module.in_keys) + [VIC_EFFERENCE_KEY]
    m.in_keys = list(m.in_keys) + [VIC_EFFERENCE_KEY]
    return cfg
