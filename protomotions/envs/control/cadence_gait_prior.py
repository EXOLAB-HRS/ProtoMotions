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
"""Human gait-angle prior for teacher_cadence_v1 candidates (2026-10-06).

CadenceGaitPercentSteering adds, per leg, the gait percent since that foot's last stance onset
(the same ankle/toe height rule as the gait clock; in r8 it matches contact-force heel strike
within one 30 Hz frame) divided by that leg's previous stride time. It does not change the
command, the clock or any existing context field.

human_gait_prior rewards knee flexion and ankle dorsiflexion close to a human table
(mean and SD by speed and gait percent; scripts/build_human_gait_table.py, Fukuchi et al. 2018):
r = exp(-0.5 * mean_j ((q_j - mu_j) / (k * max(sd_j, sd_floor)))^2), averaged over both legs
(k = kernel_scale, 1 in r10);
1 where the gait percent is not yet valid (first stride after reset, stop commands).
"""
from dataclasses import dataclass

import torch
from torch import Tensor

from protomotions.envs.context_paths import FieldPath
from protomotions.envs.control.cadence_steering import CadencePhaseSteering, CadencePhaseSteeringConfig


def steering_path(name):
    """FieldPath for a steering-context field that is not declared on SteeringContext."""
    path = FieldPath("steering")
    path.name = name
    return path


GAIT_PCT = steering_path("gait_pct")
GAIT_PCT_VALID = steering_path("gait_pct_valid")


@dataclass
class CadenceGaitPercentSteeringConfig(CadencePhaseSteeringConfig):
    _target_: str = "protomotions.envs.control.cadence_gait_prior.CadenceGaitPercentSteering"
    # A stride longer than this (s) or a percent above max_pct marks the leg invalid (pause, stop).
    max_stride_s: float = 2.0
    max_pct: float = 1.2


class CadenceGaitPercentSteering(CadencePhaseSteering):
    def __init__(self, config, env):
        super().__init__(config, env)
        n, device = len(self._tar_speed), self._tar_speed.device
        self._leg_onset = torch.zeros(n, 2, device=device)
        self._leg_stride = torch.ones(n, 2, device=device)
        self._leg_strides = torch.zeros(n, 2, dtype=torch.long, device=device)
        self._prev_stance = torch.zeros(n, 2, dtype=torch.bool, device=device)

    def reset(self, env_ids):
        super().reset(env_ids)
        if len(env_ids):
            self._leg_onset[env_ids] = self.env.progress_buf[env_ids].float()[:, None]
            self._leg_stride[env_ids] = 1.
            self._leg_strides[env_ids] = 0
            self._prev_stance[env_ids] = False

    def update_gait(self):
        super().update_gait()
        now = self.env.progress_buf.float()[:, None].expand(-1, 2)
        onset = self._stance & ~self._prev_stance
        self._prev_stance[:] = self._stance
        self._leg_stride = torch.where(onset & (self._leg_strides >= 0), now - self._leg_onset, self._leg_stride)
        self._leg_strides += onset.long()
        self._leg_onset = torch.where(onset, now, self._leg_onset)

    def populate_context(self, ctx):
        super().populate_context(ctx)
        now = self.env.progress_buf.float()[:, None]
        pct = (now - self._leg_onset) / self._leg_stride.clamp_min(1.)
        stride_s = self._leg_stride * self.env.dt
        walking = (self._tar_speed >= self.config.cadence_min_speed)[:, None]
        ctx.steering.gait_pct = pct
        ctx.steering.gait_pct_valid = (walking & (self._leg_strides >= 2) & (pct <= self.config.max_pct)
                                       & (stride_s <= self.config.max_stride_s))


_TABLES = {}


def _table(path, device):
    key = (path, str(device))
    if key not in _TABLES:
        t = torch.load(path, map_location=device, weights_only=False)
        if list(t["joints"]) != ["Knee", "Ankle"]:
            raise ValueError(f"{path}: joints {t['joints']} != ['Knee', 'Ankle']")
        _TABLES[key] = (t["speeds"].to(device), t["mean"].to(device), t["sd"].to(device))
    return _TABLES[key]


def human_gait_prior(dof_pos: Tensor, tar_speed: Tensor, gait_pct: Tensor, gait_pct_valid: Tensor,
                     table_file: str, left_dofs: list, right_dofs: list, signs: list,
                     sd_floor_deg: float = 3.0, kernel_scale: float = 1.0,
                     tar_cadence_ratio: Tensor = None, ratio_tol: float = 0.0,
                     terminal_knee_weight: float = 1.0, terminal_pct: tuple = (8, 88)) -> Tensor:
    """left_dofs/right_dofs: [knee, ankle] dof indices; signs map dof (rad) to flexion/dorsiflexion.

    kernel_scale widens the kernel (error divided by kernel_scale * sd). At 1 (r10), a 20 deg knee
    error against a 5 deg SD gives exp(-8) ~ 0 and no gradient where the error is largest.
    """
    speeds, mean, sd = _table(table_file, dof_pos.device)
    s = torch.bucketize(tar_speed, (speeds[1:] + speeds[:-1]) / 2).clamp(0, len(speeds) - 1)
    sign = torch.tensor(signs, device=dof_pos.device, dtype=dof_pos.dtype)
    rewards = []
    for leg, dofs in enumerate((left_dofs, right_dofs)):
        p = (gait_pct[:, leg].clamp(0, 1) * 100).round().long()
        q = torch.rad2deg(dof_pos[:, dofs]) * sign
        z = (q - mean[s, p]) / (kernel_scale * sd[s, p].clamp_min(sd_floor_deg))
        terminal = (p < terminal_pct[0]) | (p > terminal_pct[1])
        w = torch.ones_like(z)
        w[:, 0] = torch.where(terminal, torch.full_like(z[:, 0], terminal_knee_weight), w[:, 0])
        r = torch.exp(-0.5 * (w * z ** 2).sum(-1) / w.sum(-1))
        if ratio_tol > 0 and tar_cadence_ratio is not None:
            r = torch.where((tar_cadence_ratio - 1).abs() < ratio_tol, r, torch.ones_like(r))
        rewards.append(torch.where(gait_pct_valid[:, leg], r, torch.ones_like(r)))
    return (rewards[0] + rewards[1]) / 2
