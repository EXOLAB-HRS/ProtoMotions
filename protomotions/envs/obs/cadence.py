"""Cadence-condition observations for the c-a7-cadence candidate.

The same key ("cadence_cond") conditions the actor, critic and AMP discriminator.
Agent samples carry the commanded ratio rho*; expert samples carry the reference
frame's label from scripts/cadence_labels.py.
"""
from pathlib import Path

import torch
from torch import Tensor

_LABELS = {}


def compute_cadence_cond_obs(tar_cadence_ratio: Tensor) -> Tensor:
    return tar_cadence_ratio[:, None]


def compute_cadence_cond_from_motion_lib(motion_lib, motion_ids: Tensor, motion_times: Tensor,
                                         dt: float, label_file: str) -> Tensor:
    key = (label_file, str(motion_ids.device))
    if key not in _LABELS:
        data = torch.load(Path(label_file), map_location="cpu", weights_only=False)
        total = int(motion_lib.length_starts[-1] + motion_lib.motion_num_frames[-1])
        if data["pack_frames"] != total:
            raise RuntimeError(f"Cadence labels cover {data['pack_frames']} frames, motion lib has {total}")
        _LABELS[key] = data["labels"].to(motion_ids.device)
    frames = motion_lib.motion_num_frames[motion_ids]
    local = (motion_times / motion_lib.motion_dt[motion_ids]).round().long()
    local = torch.minimum(local.clamp_min(0), frames - 1)
    return _LABELS[key][motion_lib.length_starts[motion_ids] + local][:, None]


def compute_cadence_phase_obs(cadence_phase: Tensor) -> Tensor:
    """teacher_cadence_v1 gait clock [sin phi, cos phi]; zeros while the clock is inactive."""
    return cadence_phase
