"""teacher_se_v1 s5amp (2026-10-26, candidate run recipe): s4pelvis (s3 + pelvis_gait_prior 0.1) with the AMP
discriminator restricted to a body subset (protomotions/envs/obs/disc_body_subset.py).

Why: in every teacher_se_v1 / vic_TC run the discriminator separates agent from motion-pack frames almost perfectly
(agent_acc ~0.98, pos_acc 1.0, AMP reward ~0.03 vs a 0.4 share), so the style term gives the policy little signal.
Earlier single-knob screens did not move it: grad penalty 5 -> 20 (output/260922/e07_a7_gp20_amp_saturation:
1.000 / 0.978) and disc lr 5e-5 -> 2e-5 (a5_disclr2e5: 1.000 / 0.981). On se-v1-s3 steady rollouts vs the motion pack
the largest distribution gaps are the passive joints (Elbow_x/y, Hand_x/y/z have kp = 0 and effort 0): joint-velocity
SD 11-82 x the pack's, mean offsets z 2-9, plus wrist offsets. The discriminator sees every body (26) for 8 steps.
This recipe keeps the plant and every reward and only changes what the discriminator sees:
  DISC_BODIES = "legs": Pelvis, both legs (Hip, Knee, Talus, Ankle, Toe), Torso, Spine, Chest (14 bodies)
  DISC_BODIES = "prox": legs + L/R Thorax and Shoulder (upper arms, 18 bodies)
The discriminator's first layer and input normalizer are cut to the kept columns by the launcher
(teacher_followup.py subset_disc_inputs), so it starts from the s3 discriminator on those inputs.
"""
import dataclasses

from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.obs.disc_body_subset import (
    compute_historical_max_coords_subset_from_motion_lib, compute_historical_max_coords_subset_from_state)

from examples.experiments.steering.mlp_human_model_v3_teacher_se_v1_s4pelvis import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_teacher_se_v1_s4pelvis import env_config as _s4_env_config
from examples.experiments.steering.mlp_human_model_v3_teacher_se_v1_s3 import agent_config as _s3_agent_config

BODY_SETS = {"legs": list(range(14)), "prox": list(range(14)) + [16, 17, 21, 22]}
DISC_BODIES = "legs"
DISC_KEY = "disc_coords_obs"


def env_config(robot_cfg, args):
    cfg = _s4_env_config(robot_cfg, args)
    hist = EnvContext.historical
    cfg.observation_components[DISC_KEY] = MdpComponent(
        compute_func=compute_historical_max_coords_subset_from_state,
        dynamic_vars={"historical_rigid_body_pos": hist.rigid_body_pos, "historical_rigid_body_rot": hist.rigid_body_rot,
                      "historical_rigid_body_vel": hist.rigid_body_vel,
                      "historical_rigid_body_ang_vel": hist.rigid_body_ang_vel,
                      "historical_ground_heights": hist.ground_heights, "historical_body_contacts": hist.body_contacts},
        static_params={"body_ids": BODY_SETS[DISC_BODIES], "local_obs": True, "root_height_obs": True, "w_last": True})
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    cfg = _s3_agent_config(robot_cfg, env_cfg, args)
    m = cfg.model
    for module in (m.discriminator, m.discriminator.models[0]):
        module.in_keys = [DISC_KEY if k == "historical_max_coords_obs" else k for k in module.in_keys]
    if DISC_KEY not in m.in_keys:
        m.in_keys = list(m.in_keys) + [DISC_KEY]
    old = cfg.reference_obs_components["historical_max_coords_obs"]
    cfg.reference_obs_components[DISC_KEY] = MdpComponent(
        compute_func=compute_historical_max_coords_subset_from_motion_lib, dynamic_vars={},
        static_params={**old.get_params(), "body_ids": BODY_SETS[DISC_BODIES]})
    return cfg
