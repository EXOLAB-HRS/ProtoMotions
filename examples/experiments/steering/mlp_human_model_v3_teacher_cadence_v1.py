"""teacher_cadence_v1: cadence command as a gait clock on top of the c-a7-cadence candidate.

Parent pc_240_4_r3 (transition-tails, 240 Hz implicit PD). Same rho* command, sim f0 and
cadence-conditioned discriminator as mlp_human_model_v3_c_a7_cadence.py, plus:
- the actor and critic also see the gait clock [sin phi, cos phi] (CadencePhaseSteering);
- cadence_phase_contact (weight PHASE_WEIGHT, taken from heading) rewards the clock's
  stance foot on the ground during single support.
cadence_tracking is kept at weight 0 as a logged diagnostic. The c-a7-cadence run
(260930 e04) moved cadence 0.2-0.3% for a 20% rho* change with the step-period reward
alone. Warm starts append zero-weight columns (teacher_followup.py
--append-cadence-input), so the parent's behaviour is unchanged at the first step.
"""
import dataclasses

from examples.experiments.steering.mlp_human_model_v3_c_a7_cadence import *
from examples.experiments.steering.mlp_human_model_v3_c_a7_cadence import (
    env_config as _cadence_env_config, agent_config as _cadence_agent_config, KEY, CADENCE_WEIGHT)
from protomotions.envs.context_views import EnvContext
from protomotions.envs.control.cadence_steering import CadencePhaseSteeringConfig
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.obs.cadence import compute_cadence_phase_obs
from protomotions.envs.rewards.locomotion_quality import cadence_phase_contact

PHASE_KEY = "cadence_phase"
# 0.15 in the first run (261001 e01): ignoring the clock cost ~2% of the reward.
PHASE_WEIGHT = 0.30


def env_config(robot_cfg, args):
    cfg = _cadence_env_config(robot_cfg, args)
    base = cfg.control_components["steering"]
    cfg.control_components["steering"] = CadencePhaseSteeringConfig(
        **{f.name: getattr(base, f.name) for f in dataclasses.fields(base) if f.name != "_target_"})
    cfg.observation_components[PHASE_KEY] = MdpComponent(
        compute_func=compute_cadence_phase_obs,
        dynamic_vars={"cadence_phase": EnvContext.steering.cadence_phase})
    # heading 0.45 -> 0.15: 0.30 to the clock; the step-period reward stays logged at 0.
    cfg.reward_components["heading_rew"].static_params["weight"] += CADENCE_WEIGHT - PHASE_WEIGHT
    cfg.reward_components["cadence_tracking"].static_params["weight"] = 0.0
    cfg.reward_components["cadence_phase_contact"] = MdpComponent(
        compute_func=cadence_phase_contact,
        dynamic_vars={"foot_contact": EnvContext.steering.foot_contact,
                      "phase_stance_left": EnvContext.steering.phase_stance_left,
                      "phase_single_support": EnvContext.steering.phase_single_support},
        static_params={"weight": PHASE_WEIGHT})
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    cfg = _cadence_agent_config(robot_cfg, env_cfg, args)
    m = cfg.model
    # The discriminator keeps cadence_cond only: reference frames have no clock.
    for module in (m, m.actor, m.actor.mu_model, m.critic):
        module.in_keys = list(module.in_keys) + [PHASE_KEY]
    return cfg
