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
"""Joint moments in the AMP discriminator (vic4 tqamp candidate, 2026-11-06; user: "방법1 로 학습 바로 돌리자").

Why: the moment reward (vic_moment_prior) and the body-motion discriminator pull the gait apart. More reward weight or a
longer run moves torque toward human but W2 collapses (server 244: push-off 1.11, W2 0.54 / 0.33), and the 26-body
discriminator holds W1 / W2 while torque stays near ff100 (247). Here the discriminator sees the leg moments next to the
body motion, so torque and kinematics are matched as one distribution.

Agent samples: the plant's total sagittal leg torque (PD + feed-forward + passive) in N m/kg from
VicSteering (v4_moment_disc), [N, 6] = L hip / knee / ankle, R hip / knee / ankle, plant signs.
Expert samples: the motion pack has no torque, so each reference frame gets the human moment of its gait percent and
speed from the Lencioni 2019 table (training data; the W2-2 judge uses WBDS 2018, so the judge stays independent):
mean + noise * sd * eps per joint (eps ~ N(0, 1), the between-subject spread). Gait percent per leg is built from the
reference foot heights with the env rule (stance = any foot body below its stance height, onset = stance start):
onsets closer than min_gap_s are merged (contact flicker); between two onsets up to max_stride_s apart the percent is
linear; outside them (clip start / end, pauses) it is extrapolated from the nearest stride (or the clip median, or
the other leg + 50 %) modulo 1. Speed is the 1 s mean of the root horizontal speed, nearest table speed.
"""
import math

import torch
from torch import Tensor

_LABELS = {}
_TABLES = {}


def compute_disc_moment_obs(vic_moment_feat: Tensor) -> Tensor:
    return vic_moment_feat


def _leg_percent(h: Tensor, heights: Tensor, dt: float, min_gap_s: float, max_stride_s: float, default_stride_s: float):
    """Gait percent [T] of one leg from foot body heights h [T, F]; None if the leg has no stance onset."""
    st = (h < heights).any(-1)
    on = ((st[1:] & ~st[:-1]).nonzero().flatten() + 1).tolist()
    keep = []
    for o in on:
        if not keep or (o - keep[-1]) * dt >= min_gap_s:
            keep.append(o)
    if not keep:
        return None
    T = h.shape[0]
    strides = [b - a for a, b in zip(keep[:-1], keep[1:]) if (b - a) * dt <= max_stride_s]
    med = float(torch.tensor(strides, dtype=torch.float).median()) if strides else default_stride_s / dt
    pct = torch.empty(T)
    t = torch.arange(T, dtype=torch.float)
    ks = torch.tensor(keep, dtype=torch.float)
    for i in range(T):
        j = int(torch.searchsorted(ks, t[i], right=True)) - 1          # last onset <= t
        if 0 <= j < len(keep) - 1 and (keep[j + 1] - keep[j]) * dt <= max_stride_s:
            pct[i] = (t[i] - keep[j]) / (keep[j + 1] - keep[j])
        elif j >= 0:
            prev = keep[j] - keep[j - 1] if j >= 1 and (keep[j] - keep[j - 1]) * dt <= max_stride_s else med
            pct[i] = ((t[i] - keep[j]) / prev) % 1.0
        else:
            nxt = keep[1] - keep[0] if len(keep) > 1 and (keep[1] - keep[0]) * dt <= max_stride_s else med
            pct[i] = (1.0 - (keep[0] - t[i]) / nxt) % 1.0
    return pct


def reference_gait_labels(motion_lib, feet, stance_heights, min_gap_s=0.5, max_stride_s=2.0, default_stride_s=1.2,
                          speed_window_s=1.0):
    """Per pack frame: gait percent [F, 2] (L, R) and root horizontal speed [F] (cached per motion lib)."""
    key = (id(motion_lib), tuple(map(tuple, feet)), tuple(stance_heights))
    if key in _LABELS:
        return _LABELS[key]
    gts, gvs = motion_lib.gts.cpu(), motion_lib.gvs.cpu()
    starts, frames, dts = (motion_lib.length_starts.cpu(), motion_lib.motion_num_frames.cpu(), motion_lib.motion_dt.cpu())
    heights = torch.tensor(stance_heights, dtype=torch.float)
    total = int(starts[-1] + frames[-1])
    pct = torch.zeros(total, 2)
    speed = torch.zeros(total)
    for m in range(len(frames)):
        a, n, dt = int(starts[m]), int(frames[m]), float(dts[m])
        h = gts[a:a + n, :, 2]
        legs = [_leg_percent(h[:, f], heights, dt, min_gap_s, max_stride_s, default_stride_s) for f in feet]
        for leg in range(2):
            p = legs[leg]
            if p is None:
                other = legs[1 - leg]
                p = (other + 0.5) % 1.0 if other is not None else torch.full((n,), 0.3)
            pct[a:a + n, leg] = p
        v = gvs[a:a + n, 0, :2].norm(dim=-1)
        w = max(1, int(round(speed_window_s / dt)))
        k = torch.ones(1, 1, w) / w
        vp = torch.nn.functional.pad(v[None, None], (w // 2, w - 1 - w // 2), mode="replicate")
        speed[a:a + n] = torch.nn.functional.conv1d(vp, k)[0, 0]
    _LABELS[key] = (pct.to(motion_lib.gts.device), speed.to(motion_lib.gts.device))
    return _LABELS[key]


def compute_disc_moment_from_motion_lib(motion_lib, motion_ids: Tensor, motion_times: Tensor, dt: float,
                                        table_file: str, signs, feet, stance_heights, noise: float = 1.0,
                                        sd_floor: float = 0.1) -> Tensor:
    device = motion_ids.device
    pct, speed = reference_gait_labels(motion_lib, feet, stance_heights)
    key = (table_file, tuple(signs), str(device))
    if key not in _TABLES:
        t = torch.load(table_file, map_location=device, weights_only=False)
        sg = torch.tensor(signs, device=device, dtype=torch.float)
        _TABLES[key] = (t["speeds"].float().to(device), t["mean"].float().to(device) * sg, t["sd"].float().to(device))
    speeds, mean, sd = _TABLES[key]
    frames = motion_lib.motion_num_frames[motion_ids]
    local = ((motion_times - dt).clamp_min(0) / motion_lib.motion_dt[motion_ids]).round().long()   # the latest history step
    idx = motion_lib.length_starts[motion_ids] + torch.minimum(local, frames - 1)
    s = torch.bucketize(speed[idx], (speeds[1:] + speeds[:-1]) / 2).clamp(0, len(speeds) - 1)
    out = []
    for leg in range(2):
        p = (pct[idx, leg].clamp(0, 1) * (mean.shape[1] - 1)).round().long()
        m, d = mean[s, p], sd[s, p].clamp_min(sd_floor)
        out.append(m + noise * d * torch.randn_like(m))
    return torch.cat(out, -1)                                                                      # [N, 6]
