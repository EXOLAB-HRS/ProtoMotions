"""teacher_cadence_v1 revision 3: wider rho*, tighter cadence tracking, signed clock reward.

r2 (261001 e02) moved cadence 0.3-0.5% for rho* 0.9 -> 1.1: the parent's natural gait
already earned 0.82 of the 0/1 clock reward and ~0.99 of the 10%-wide tracking reward.
Changes from mlp_human_model_v3_teacher_cadence_v1.py:
- rho* ~ U[0.8, 1.25] (30% at 1); discriminator labels clamped to the same range
  (forward25_cafb4b0d_cadence_labels_r080_125.pt, reference p10/p90 0.886/1.221);
- cadence_tracking back on: weight 0.12, relative scale 0.04;
- clock reward cadence_phase_contact_signed (+1 / 0 / -1), weight 0.18.
heading_rew keeps r2's 0.15, so the cadence share stays 0.30.
Warm start: teacher_followup.py --append-cadence-input with r3 normalizer stats.
"""
import hashlib
from pathlib import Path

import torch

from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1 import *
from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1 import (
    env_config as _v1_env_config, agent_config as _v1_agent_config, KEY)
from protomotions.envs.rewards.locomotion_quality import cadence_phase_contact_signed

R3_LABEL_FILE = Path(__file__).resolve().parents[4] / "share/cadence/forward25_cafb4b0d_cadence_labels_r080_125.pt"
RATIO_RANGE = (0.8, 1.25)
TRACKING_WEIGHT, TRACKING_SCALE = 0.12, 0.04
PHASE_WEIGHT_R3 = 0.18


def env_config(robot_cfg, args):
    labels = torch.load(R3_LABEL_FILE, map_location="cpu", weights_only=False)
    if tuple(labels["rho_range"]) != RATIO_RANGE:
        raise RuntimeError(f"{R3_LABEL_FILE.name}: rho_range {labels['rho_range']} != {RATIO_RANGE}")
    if hashlib.sha256(Path(args.motion_file).read_bytes()).hexdigest() != labels["pack_sha256"]:
        raise RuntimeError(f"Cadence labels were built for pack {labels['pack_sha256'][:8]}")
    cfg = _v1_env_config(robot_cfg, args)
    steering = cfg.control_components["steering"]
    steering.cadence_ratio_min, steering.cadence_ratio_max = RATIO_RANGE
    tracking = cfg.reward_components["cadence_tracking"].static_params
    tracking["weight"], tracking["relative_scale"] = TRACKING_WEIGHT, TRACKING_SCALE
    phase = cfg.reward_components["cadence_phase_contact"]
    phase.compute_func = cadence_phase_contact_signed
    phase.static_params["weight"] = PHASE_WEIGHT_R3
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    cfg = _v1_agent_config(robot_cfg, env_cfg, args)
    cfg.reference_obs_components[KEY].static_params["label_file"] = str(R3_LABEL_FILE)
    return cfg
