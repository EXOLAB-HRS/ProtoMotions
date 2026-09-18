# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Retarget packaged SMPL motions onto the human_model_v2 coordinate set.

The source library stores 24 bodies with three exponential-map coordinates per
joint. v2 keeps the same morphology but replaces the leg chain with the fixed
MyoLeg axes (hip 3, knee 1, ankle 1, subtalar 1, MTP 1) and inserts a talus
body, so a source clip cannot be replayed directly.

Two stages, both differentiable and batched over frames:

1. Joint-rotation matching. Every v2 coordinate is fitted so the composed hinge
   rotation reproduces the source joint rotation. The ankle target is matched
   against the talus/calcaneus product because the source lumps both into one
   ball joint. Coordinates are parameterised inside their ROM, so the fit
   reports how much of the source pose the physiological limits reject instead
   of silently exceeding them.
2. Leg refinement. Warm-started from stage 1, the 14 leg coordinates also carry
   world-position targets for knee/ankle/toe and a foot-orientation target,
   plus a second-difference smoothness term, so ROM-clipped frames distribute
   their residual over the chain rather than breaking foot placement.

The result is written as a MotionLib ``.pt`` **without** ``lrs``: the library's
local-rotation interpolation path assumes three coordinates per body and would
reject the 59-coordinate model.

This is a reference-motion converter. It performs no teacher training and makes
no claim that the converted clips are biomechanically valid human motion.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from protomotions.components.pose_lib import (
    KinematicInfo,
    extract_kinematic_info,
    extract_transforms_from_qpos_non_root_ignore_fixed_helper,
    fk_batch_mjcf_with_velocities,
)
from protomotions.utils.rotations import matrix_to_quaternion, quaternion_to_matrix

from ...common.paths import PACKAGE_ROOT

log = logging.getLogger(__name__)

SOURCE_MJCF = PACKAGE_ROOT / "human_model_v1/assets/smpl_humanoid.xml"
TARGET_MJCF = PACKAGE_ROOT / "human_model_v2/assets/human_model_v2.xml"

SIDES = ("L", "R")
ANKLE_SOURCE_BODY = "{side}_Ankle"
# v2 splits the source ankle ball joint into talus (ankle_angle) and
# calcaneus (subtalar_angle), so one source rotation drives two v2 bodies.
TALUS_BODY = "{side}_Talus"
CALCANEUS_BODY = "{side}_Ankle"

# v2 inherits the source link offsets, so the source ankle body coincides with
# the v2 talus rather than with the v2 body that kept the ankle name.
POSITION_TARGETS = (
    ("{side}_Knee", "{side}_Knee"),
    ("{side}_Ankle", "{side}_Talus"),
    ("{side}_Toe", "{side}_Toe"),
)

_MOTION_META_FIELDS = ("motion_lengths", "motion_dt", "motion_num_frames", "motion_weights")


@dataclass(frozen=True)
class RetargetConfig:
    rotation_steps: int = 400
    rotation_lr: float = 0.05
    leg_steps: int = 600
    leg_lr: float = 0.01
    position_weight: float = 200.0
    foot_rotation_weight: float = 5.0
    smoothness_weight: float = 0.5
    ground_align: bool = True
    ground_align_tolerance_m: float = 0.002
    device: str = "cpu"


def _body_index(info: KinematicInfo, name: str) -> int:
    return info.body_names.index(name)


def _movable_body_order(info: KinematicInfo) -> List[int]:
    return list(info.hinge_axes_map.keys())


def _coordinate_slices(info: KinematicInfo) -> Dict[int, slice]:
    slices: Dict[int, slice] = {}
    start = 0
    for body_idx, axes in info.hinge_axes_map.items():
        slices[body_idx] = slice(start, start + len(axes))
        start += len(axes)
    return slices


def _bounded(params: Tensor, lower: Tensor, upper: Tensor) -> Tensor:
    """Sigmoid parameterisation keeps every coordinate inside its ROM box."""
    return lower + (upper - lower) * torch.sigmoid(params)


def _inverse_bounded(values: Tensor, lower: Tensor, upper: Tensor) -> Tensor:
    normalized = ((values - lower) / (upper - lower)).clamp(1e-4, 1 - 1e-4)
    return torch.log(normalized / (1 - normalized))


def _compose_body_rotations(info: KinematicInfo, coords: Tensor) -> Tensor:
    return extract_transforms_from_qpos_non_root_ignore_fixed_helper(
        info.hinge_axes_map, coords, qpos_is_exp_map_on_3dof_joints=False
    )


def _source_rotation_targets(
    source_info: KinematicInfo,
    target_info: KinematicInfo,
    local_rotations: Tensor,
) -> Tuple[Tensor, List[int], List[Tuple[int, int]]]:
    """Targets in movable-body order, with the ankle pairs kept separate.

    Returns the stacked targets, the positions matched one-to-one, and the
    (talus, calcaneus) position pairs whose product carries the source ankle.
    """
    movable = _movable_body_order(target_info)
    position_of = {body_idx: i for i, body_idx in enumerate(movable)}

    targets = torch.eye(3, dtype=local_rotations.dtype, device=local_rotations.device)
    targets = targets.view(1, 1, 3, 3).repeat(local_rotations.shape[0], len(movable), 1, 1)

    direct: List[int] = []
    fused: List[Tuple[int, int]] = []
    fused_names = {TALUS_BODY.format(side=s) for s in SIDES} | {
        CALCANEUS_BODY.format(side=s) for s in SIDES
    }

    for body_idx in movable:
        name = target_info.body_names[body_idx]
        if name in fused_names:
            continue
        targets[:, position_of[body_idx]] = local_rotations[:, _body_index(source_info, name)]
        direct.append(position_of[body_idx])

    for side in SIDES:
        talus = position_of[_body_index(target_info, TALUS_BODY.format(side=side))]
        calcaneus = position_of[_body_index(target_info, CALCANEUS_BODY.format(side=side))]
        targets[:, talus] = local_rotations[
            :, _body_index(source_info, ANKLE_SOURCE_BODY.format(side=side))
        ]
        fused.append((talus, calcaneus))

    return targets, direct, fused


def _rotation_loss(
    predicted: Tensor,
    targets: Tensor,
    direct: Sequence[int],
    fused: Sequence[Tuple[int, int]],
) -> Tensor:
    direct_idx = torch.tensor(list(direct), dtype=torch.long, device=predicted.device)
    loss = (predicted[:, direct_idx] - targets[:, direct_idx]).pow(2).sum(dim=(-1, -2)).mean()
    for talus, calcaneus in fused:
        composed = torch.matmul(predicted[:, talus], predicted[:, calcaneus])
        loss = loss + (composed - targets[:, talus]).pow(2).sum(dim=(-1, -2)).mean()
    return loss


def _forward_kinematics(
    info: KinematicInfo, root_pos: Tensor, root_rot_mat: Tensor, coords: Tensor
) -> Tuple[Tensor, Tensor]:
    """Differentiable FK.

    ``pose_lib.compute_forward_kinematics_from_transforms`` writes body slices
    in place, so its saved views are invalidated before backward runs. This
    accumulates into Python lists instead, which keeps the graph intact.
    """
    composed = _compose_body_rotations(info, coords)
    slot_of = {body_idx: position for position, body_idx in enumerate(_movable_body_order(info))}
    local_pos = info.local_pos.to(device=coords.device, dtype=coords.dtype)
    local_rot_ref = info.local_rot_ref_mat.to(device=coords.device, dtype=coords.dtype)

    world_pos: List[Tensor] = [root_pos]
    world_rot: List[Tensor] = [root_rot_mat]
    for body in range(1, info.num_bodies):
        parent = int(info.parent_indices[body])
        if parent >= body:
            raise ValueError(f"body {body} precedes its parent {parent}; FK order broken")
        parent_rot = world_rot[parent]
        effective = local_rot_ref[body]
        if body in slot_of:
            effective = torch.matmul(effective, composed[:, slot_of[body]])
        world_rot.append(torch.matmul(parent_rot, effective))
        world_pos.append(
            world_pos[parent] + torch.matmul(parent_rot, local_pos[body].view(3, 1)).squeeze(-1)
        )
    return torch.stack(world_pos, dim=1), torch.stack(world_rot, dim=1)


def _leg_coordinate_indices(info: KinematicInfo) -> Tensor:
    slices = _coordinate_slices(info)
    indices: List[int] = []
    for side in SIDES:
        for suffix in ("Hip", "Knee", "Talus", "Ankle", "Toe"):
            body_idx = _body_index(info, f"{side}_{suffix}")
            if body_idx in slices:
                indices.extend(range(slices[body_idx].start, slices[body_idx].stop))
    return torch.tensor(sorted(indices), dtype=torch.long)


def _fit_rotations(
    source_info: KinematicInfo,
    target_info: KinematicInfo,
    local_rotations: Tensor,
    config: RetargetConfig,
) -> Tuple[Tensor, Dict[str, float]]:
    device = local_rotations.device
    dtype = local_rotations.dtype
    lower = target_info.dof_limits_lower.to(device=device, dtype=dtype)
    upper = target_info.dof_limits_upper.to(device=device, dtype=dtype)
    targets, direct, fused = _source_rotation_targets(source_info, target_info, local_rotations)

    frames = local_rotations.shape[0]
    neutral = torch.zeros(frames, lower.numel(), dtype=dtype, device=device)
    params = _inverse_bounded(neutral.clamp(lower, upper), lower, upper).requires_grad_(True)
    optimizer = torch.optim.Adam([params], lr=config.rotation_lr)

    for _ in range(config.rotation_steps):
        optimizer.zero_grad(set_to_none=True)
        coords = _bounded(params, lower, upper)
        loss = _rotation_loss(_compose_body_rotations(target_info, coords), targets, direct, fused)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        coords = _bounded(params, lower, upper)
        predicted = _compose_body_rotations(target_info, coords)
        residual = float(_rotation_loss(predicted, targets, direct, fused))
        per_body: Dict[str, float] = {}
        for position, body_idx in enumerate(_movable_body_order(target_info)):
            error = (predicted[:, position] - targets[:, position]).pow(2).sum(dim=(-1, -2))
            per_body[target_info.body_names[body_idx]] = float(error.max().sqrt())

    stats = {"rotation_residual": residual}
    stats.update({f"rot_max_{name}": value for name, value in per_body.items()})
    return coords.detach(), stats


def _refine_legs(
    source_info: KinematicInfo,
    target_info: KinematicInfo,
    coords: Tensor,
    root_pos: Tensor,
    root_rot_mat: Tensor,
    source_positions: Tensor,
    source_rotations: Tensor,
    config: RetargetConfig,
) -> Tensor:
    device = coords.device
    dtype = coords.dtype
    lower = target_info.dof_limits_lower.to(device=device, dtype=dtype)
    upper = target_info.dof_limits_upper.to(device=device, dtype=dtype)
    leg_idx = _leg_coordinate_indices(target_info).to(device)

    fixed = _inverse_bounded(coords, lower, upper).detach()
    free = fixed[:, leg_idx].clone().requires_grad_(True)

    pairs = [
        (
            _body_index(source_info, source.format(side=side)),
            _body_index(target_info, target.format(side=side)),
        )
        for side in SIDES
        for source, target in POSITION_TARGETS
    ]
    source_ids = torch.tensor([pair[0] for pair in pairs], dtype=torch.long, device=device)
    target_ids = torch.tensor([pair[1] for pair in pairs], dtype=torch.long, device=device)
    foot_source = torch.tensor(
        [_body_index(source_info, ANKLE_SOURCE_BODY.format(side=s)) for s in SIDES],
        dtype=torch.long,
        device=device,
    )
    foot_target = torch.tensor(
        [_body_index(target_info, CALCANEUS_BODY.format(side=s)) for s in SIDES],
        dtype=torch.long,
        device=device,
    )

    optimizer = torch.optim.Adam([free], lr=config.leg_lr)
    for _ in range(config.leg_steps):
        optimizer.zero_grad(set_to_none=True)
        current = fixed.index_copy(1, leg_idx, free)
        angles = _bounded(current, lower, upper)
        world_pos, world_rot = _forward_kinematics(target_info, root_pos, root_rot_mat, angles)

        position_error = (
            (world_pos[:, target_ids] - source_positions[:, source_ids]).pow(2).sum(-1).mean()
        )
        rotation_error = (
            (world_rot[:, foot_target] - source_rotations[:, foot_source])
            .pow(2)
            .sum((-1, -2))
            .mean()
        )
        leg_angles = angles[:, leg_idx]
        if leg_angles.shape[0] > 2:
            curvature = leg_angles[2:] - 2 * leg_angles[1:-1] + leg_angles[:-2]
            smooth_error = curvature.pow(2).mean()
        else:
            smooth_error = torch.zeros((), dtype=dtype, device=device)

        loss = (
            config.position_weight * position_error
            + config.foot_rotation_weight * rotation_error
            + config.smoothness_weight * smooth_error
        )
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        return _bounded(fixed.index_copy(1, leg_idx, free), lower, upper).detach()


def _central_difference(values: Tensor, fps: float) -> Tensor:
    if values.shape[0] < 2:
        return torch.zeros_like(values)
    forward = torch.cat([values[1:], values[-1:]], dim=0)
    backward = torch.cat([values[:1], values[:-1]], dim=0)
    span = torch.full((values.shape[0],), 2.0 / fps, dtype=values.dtype, device=values.device)
    span[0] = 1.0 / fps
    span[-1] = 1.0 / fps
    return (forward - backward) / span.view(-1, *([1] * (values.dim() - 1)))


def _foot_body_ids(info: KinematicInfo, parts: Sequence[str], device: torch.device) -> Tensor:
    return torch.tensor(
        [_body_index(info, f"{side}_{part}") for side in SIDES for part in parts],
        dtype=torch.long,
        device=device,
    )


def _ground_offset(
    source_positions: Tensor,
    world_pos: Tensor,
    source_info: KinematicInfo,
    target_info: KinematicInfo,
) -> float:
    """Lift the clip so its lowest foot sample sits where the source's does.

    A per-frame median offset leaves the deepest frames below the source floor,
    which shows up as ground penetration once the clip is replayed.
    """
    source_low = source_positions[:, _foot_body_ids(source_info, ("Ankle", "Toe"), world_pos.device), 2]
    target_low = world_pos[:, _foot_body_ids(target_info, ("Talus", "Toe"), world_pos.device), 2]
    return float(source_low.min() - target_low.min())


def retarget_motion_library(
    source_path: Path,
    output_path: Path,
    report_path: Optional[Path] = None,
    config: RetargetConfig = RetargetConfig(),
    motion_limit: Optional[int] = None,
) -> Dict:
    """Convert a packaged SMPL motion library to human_model_v2 coordinates."""
    device = torch.device(config.device)
    source_info = extract_kinematic_info(str(SOURCE_MJCF)).to(device, torch.float32)
    target_info = extract_kinematic_info(str(TARGET_MJCF)).to(device, torch.float32)

    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    if "lrs" not in payload:
        raise ValueError("source library has no local rotations; cannot retarget")
    if len(source_info.body_names) != payload["gts"].shape[1]:
        raise ValueError(
            f"source library has {payload['gts'].shape[1]} bodies but the SMPL MJCF "
            f"declares {len(source_info.body_names)}"
        )

    starts = payload["length_starts"].tolist()
    counts = payload["motion_num_frames"].tolist()
    dts = payload["motion_dt"].tolist()
    num_motions = len(counts) if motion_limit is None else min(motion_limit, len(counts))

    collected: Dict[str, List[Tensor]] = {
        key: [] for key in ("gts", "grs", "gvs", "gavs", "dps", "dvs", "contacts")
    }
    motion_reports: List[Dict] = []

    for motion in range(num_motions):
        frames = counts[motion]
        window = slice(starts[motion], starts[motion] + frames)
        fps = 1.0 / dts[motion]

        local_rot = quaternion_to_matrix(
            payload["lrs"][window].to(device).reshape(-1, 4), w_last=True
        ).reshape(frames, -1, 3, 3)
        source_pos = payload["gts"][window].to(device)
        source_rot = quaternion_to_matrix(
            payload["grs"][window].to(device).reshape(-1, 4), w_last=True
        ).reshape(frames, -1, 3, 3)
        root_pos = source_pos[:, 0].clone()
        root_rot = local_rot[:, 0]

        coords, rotation_stats = _fit_rotations(source_info, target_info, local_rot, config)
        coords = _refine_legs(
            source_info,
            target_info,
            coords,
            root_pos,
            root_rot,
            source_pos,
            source_rot,
            config,
        )

        world_pos, _ = _forward_kinematics(target_info, root_pos, root_rot, coords)
        offset = _ground_offset(source_pos, world_pos, source_info, target_info)
        if config.ground_align and abs(offset) > config.ground_align_tolerance_m:
            root_pos = root_pos.clone()
            root_pos[:, 2] += offset
        else:
            offset = 0.0

        qpos = torch.cat([root_pos, matrix_to_quaternion(root_rot, w_last=False), coords], dim=-1)
        state = fk_batch_mjcf_with_velocities(target_info, qpos, fps=fps, compute_velocities=True)
        body_pos, body_rot = state.rigid_body_pos, state.rigid_body_rot
        body_vel, body_ang_vel = state.rigid_body_vel, state.rigid_body_ang_vel
        if body_pos is None or body_rot is None or body_vel is None or body_ang_vel is None:
            raise RuntimeError(f"motion {motion} produced no velocities; clip has too few frames")

        collected["gts"].append(body_pos.cpu())
        collected["grs"].append(body_rot.cpu())
        collected["gvs"].append(body_vel.cpu())
        collected["gavs"].append(body_ang_vel.cpu())
        collected["dps"].append(coords.cpu())
        collected["dvs"].append(_central_difference(coords, fps).cpu())
        collected["contacts"].append(
            _map_contacts(payload["contacts"][window], source_info, target_info)
        )

        motion_reports.append(
            _motion_metrics(
                motion,
                source_info,
                target_info,
                source_pos,
                body_pos.to(device),
                body_vel.to(device),
                coords,
                fps,
                offset,
                rotation_stats,
            )
        )
        log.info("retargeted motion %d/%d (%d frames)", motion + 1, num_motions, frames)

    output: Dict[str, object] = {
        key: torch.cat(values, dim=0) for key, values in collected.items()
    }
    for field in _MOTION_META_FIELDS:
        output[field] = payload[field][:num_motions].clone()
    output["length_starts"] = _recompute_starts(payload["motion_num_frames"][:num_motions])
    output["motion_files"] = tuple(
        f"{name}#retargeted:human_model_v2" for name in payload["motion_files"][:num_motions]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)

    report = {
        "source": str(source_path),
        "output": str(output_path),
        "source_bodies": len(source_info.body_names),
        "target_bodies": len(target_info.body_names),
        "target_coordinates": int(target_info.nq - 7),
        "num_motions": num_motions,
        "num_frames": int(sum(entry["frames"] for entry in motion_reports)),
        "lrs_omitted": "MotionLib local-rotation interpolation assumes 3 coordinates per body",
        "contact_assumption": "talus copies the calcaneus contact flag; not a measured contact",
        "config": asdict(config),
        "summary": _summarise(motion_reports),
        "motions": motion_reports,
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2))
    return report


def _map_contacts(
    contacts: Tensor, source_info: KinematicInfo, target_info: KinematicInfo
) -> Tensor:
    mapped = torch.zeros(contacts.shape[0], len(target_info.body_names), dtype=contacts.dtype)
    for index, name in enumerate(target_info.body_names):
        source_name = name.replace("_Talus", "_Ankle")
        mapped[:, index] = contacts[:, _body_index(source_info, source_name)]
    return mapped


def _recompute_starts(frame_counts: Tensor) -> Tensor:
    starts = torch.zeros_like(frame_counts)
    starts[1:] = torch.cumsum(frame_counts, dim=0)[:-1]
    return starts


def _motion_metrics(
    motion_id: int,
    source_info: KinematicInfo,
    target_info: KinematicInfo,
    source_pos: Tensor,
    world_pos: Tensor,
    body_vel: Tensor,
    coords: Tensor,
    fps: float,
    ground_offset: float,
    rotation_stats: Dict[str, float],
) -> Dict:
    device = coords.device
    pairs = [
        (
            _body_index(source_info, source.format(side=side)),
            _body_index(target_info, target.format(side=side)),
        )
        for side in SIDES
        for source, target in POSITION_TARGETS
    ]
    source_ids = torch.tensor([pair[0] for pair in pairs], dtype=torch.long, device=device)
    target_ids = torch.tensor([pair[1] for pair in pairs], dtype=torch.long, device=device)

    error = (world_pos[:, target_ids] - source_pos[:, source_ids]).norm(dim=-1)
    lower = target_info.dof_limits_lower.to(device)
    upper = target_info.dof_limits_upper.to(device)
    violation = torch.maximum(lower - coords, coords - upper).clamp_min(0)

    foot_ids = _foot_body_ids(target_info, ("Talus", "Ankle", "Toe"), device)
    foot_height = world_pos[:, foot_ids, 2]
    grounded = foot_height < (foot_height.min() + 0.02)
    slip = body_vel[:, foot_ids, :2].norm(dim=-1)[grounded]

    worst_body, worst_error = max(
        ((key[len("rot_max_"):], value) for key, value in rotation_stats.items()
         if key.startswith("rot_max_")),
        key=lambda item: item[1],
    )
    return {
        "motion_id": motion_id,
        "frames": int(coords.shape[0]),
        "fps": fps,
        "position_rmse_m": float(error.pow(2).mean().sqrt()),
        "position_max_m": float(error.max()),
        "rom_violation_max_rad": float(violation.max()),
        "min_body_height_m": float(world_pos[..., 2].min()),
        "contact_foot_slip_mean_mps": float(slip.mean()) if slip.numel() else 0.0,
        "contact_foot_slip_max_mps": float(slip.max()) if slip.numel() else 0.0,
        "max_joint_speed_rad_s": float(_central_difference(coords, fps).abs().max()),
        "ground_offset_m": ground_offset,
        "rotation_residual": rotation_stats["rotation_residual"],
        "worst_rotation_body": worst_body,
        "worst_rotation_frobenius": worst_error,
    }


def _summarise(motions: Sequence[Dict]) -> Dict:
    return {
        "position_rmse_m_mean": sum(m["position_rmse_m"] for m in motions) / len(motions),
        "position_max_m": max(m["position_max_m"] for m in motions),
        "rom_violation_max_rad": max(m["rom_violation_max_rad"] for m in motions),
        "contact_foot_slip_max_mps": max(m["contact_foot_slip_max_mps"] for m in motions),
        "max_joint_speed_rad_s": max(m["max_joint_speed_rad_s"] for m in motions),
        "min_body_height_m": min(m["min_body_height_m"] for m in motions),
        "worst_rotation_frobenius": max(m["worst_rotation_frobenius"] for m in motions),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="SMPL to human_model_v2 motion retargeting")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rotation-steps", type=int, default=RetargetConfig.rotation_steps)
    parser.add_argument("--leg-steps", type=int, default=RetargetConfig.leg_steps)
    parser.add_argument("--motion-limit", type=int, default=None)
    parser.add_argument("--no-ground-align", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    config = RetargetConfig(
        rotation_steps=args.rotation_steps,
        leg_steps=args.leg_steps,
        device=args.device,
        ground_align=not args.no_ground_align,
    )
    report = retarget_motion_library(args.source, args.output, args.report, config, args.motion_limit)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
