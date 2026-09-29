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
"""Vectorized strength_caps must equal the original per-row loop (strength_caps_reference).

Covers the v3 profile under its three strength scenarios and v2, with joint angles
and velocities inside and well outside the curve domains (clamping and flags), both
flag layouts, float32 and float64.
"""
import pytest
import torch

from protomotions.robot_configs.human_model.profile import load_profile
from protomotions.robot_configs.human_model.dynamics import HumanJointModel
from protomotions.robot_configs.human_model.human_model_v3.model_config import robot_config, configure_strength_profile


def _states(model, n=4096, seed=0):
    g = torch.Generator().manual_seed(seed)
    lo, hi = model.lower.cpu(), model.upper.cpu()
    span = (hi - lo).clamp_min(1e-3)
    q = lo + (torch.rand(n, len(model.names), generator=g) * 1.6 - 0.3) * span  # 30% beyond each ROM end
    qd = (torch.rand(n, len(model.names), generator=g) * 2 - 1) * 20.0
    device = model.lower.device
    return q.to(device=device, dtype=lo.dtype), qd.to(device=device, dtype=lo.dtype)


def _compare(model, q, qd):
    for directional in (False, True):
        new = model.strength_caps(q, qd, directional_domain=directional)
        old = model.strength_caps_reference(q, qd, directional_domain=directional)
        for a, b in zip(new, old):
            assert a.shape == b.shape and a.dtype == b.dtype
            assert torch.equal(a, b), (a - b).abs().max() if a.is_floating_point() else (a ^ b).sum()


@pytest.mark.parametrize("scenario", ["low", "reference", "high"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_v3_vectorized_strength_caps_match_reference(scenario, dtype):
    robot = robot_config()
    configure_strength_profile(robot, scenario)
    model = HumanJointModel(load_profile("human_model_v3"), robot.kinematic_info.dof_names,
                            dtype=dtype, **robot.human_model_parameters)
    assert model._strength_rows or model._pose_strength_rows
    _compare(model, *_states(model))


def test_v2_vectorized_strength_caps_match_reference():
    robot = robot_config()
    model = HumanJointModel(load_profile("human_model_v2"), robot.kinematic_info.dof_names)
    _compare(model, *_states(model, seed=1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_v3_vectorized_strength_caps_match_reference_cuda():
    robot = robot_config()
    model = HumanJointModel(load_profile("human_model_v3"), robot.kinematic_info.dof_names,
                            device="cuda", **robot.human_model_parameters)
    _compare(model, *_states(model))
