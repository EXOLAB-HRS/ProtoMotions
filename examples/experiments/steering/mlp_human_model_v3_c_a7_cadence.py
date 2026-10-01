"""Stage A-C transition-tails recipe plus a cadence-ratio command (conditional AMP candidate).

Parent: pc_240_4_r3 (transition-tails, 240 Hz implicit PD). rho* in [0.9, 1.1]
(30% natural, rho* = 1) scales the parent's simulated natural cadence at the commanded
speed, f0 = F0_SLOPE * v* + F0_INTERCEPT. The first run (260930 e01) took f0 from the
reference pack, about 20% below the simulated gait, and learned no rho* response. The ratio is appended to the actor, critic, discriminator and
disc-critic inputs as "cadence_cond"; the discriminator's expert samples use the
per-frame labels of scripts/cadence_labels.py, so the prior is conditioned on
cadence instead of pulling every speed to its natural cadence. Cadence tracking
takes 0.10 of the heading weight. Warm starts append a zero input column
(teacher_followup.py --append-cadence-input), so the parent's behaviour is unchanged
at the first step.
"""
import dataclasses
import hashlib
from pathlib import Path

import torch

from examples.experiments.steering.mlp_human_model_v3_c_a7_transition_tails import *
from examples.experiments.steering.mlp_human_model_v3_a7_followups import (
    agent_config as _followup_agent_config, stage_c_transition_tails_recipe)
from protomotions.envs.context_views import EnvContext
from protomotions.envs.control.cadence_steering import CadenceSteeringConfig
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.obs.cadence import compute_cadence_cond_obs, compute_cadence_cond_from_motion_lib
from protomotions.envs.rewards.locomotion_quality import cadence_tracking

LABEL_FILE = Path(__file__).resolve().parents[4] / "share/cadence/forward25_cafb4b0d_cadence_labels.pt"
CADENCE_WEIGHT = 0.10
CADENCE_RELATIVE_SCALE = 0.10
# pc_240_4_r3 Stage A, seed 2942, 60 rollouts (output/260930/e02_c_a7_cadence_pc240_natural_local):
# f = 1.907 / 1.964 / 2.034 steps/s at v* = 0.8 / 1.0 / 1.2 m/s.
F0_SLOPE, F0_INTERCEPT = 0.318, 1.651
KEY = "cadence_cond"


def _labels():
    return torch.load(LABEL_FILE, map_location="cpu", weights_only=False)


def env_config(robot_cfg, args):
    labels = _labels()
    motion = Path(args.motion_file)
    if hashlib.sha256(motion.read_bytes()).hexdigest() != labels["pack_sha256"]:
        raise RuntimeError(f"Cadence labels were built for pack {labels['pack_sha256'][:8]}, not {motion}")
    cfg = stage_c_transition_tails_recipe(robot_cfg, args)
    base = cfg.control_components["steering"]
    cfg.control_components["steering"] = CadenceSteeringConfig(
        **{f.name: getattr(base, f.name) for f in dataclasses.fields(base) if f.name != "_target_"},
        cadence_f0_slope=F0_SLOPE, cadence_f0_intercept=F0_INTERCEPT,
        cadence_min_speed=labels["min_speed"],
        cadence_ratio_min=labels["rho_range"][0], cadence_ratio_max=labels["rho_range"][1])
    cfg.observation_components[KEY] = MdpComponent(
        compute_func=compute_cadence_cond_obs,
        dynamic_vars={"tar_cadence_ratio": EnvContext.steering.tar_cadence_ratio})
    cfg.reward_components["heading_rew"].static_params["weight"] -= CADENCE_WEIGHT
    cfg.reward_components["cadence_tracking"] = MdpComponent(
        compute_func=cadence_tracking,
        dynamic_vars={"measured_cadence": EnvContext.steering.measured_cadence,
                      "tar_cadence": EnvContext.steering.tar_cadence,
                      "cadence_valid": EnvContext.steering.cadence_valid},
        static_params={"relative_scale": CADENCE_RELATIVE_SCALE, "weight": CADENCE_WEIGHT})
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    cfg = _followup_agent_config(robot_cfg, env_cfg, args)
    m = cfg.model
    # Append last so the parent's input columns keep their positions.
    for module in (m, m.actor, m.actor.mu_model, m.critic, m.discriminator, m.discriminator.models[0],
                   m.disc_critic, m.disc_critic.models[0]):
        module.in_keys = list(module.in_keys) + [KEY]
    cfg.reference_obs_components[KEY] = MdpComponent(
        compute_func=compute_cadence_cond_from_motion_lib,
        dynamic_vars={}, static_params={"label_file": str(LABEL_FILE)})
    return cfg
