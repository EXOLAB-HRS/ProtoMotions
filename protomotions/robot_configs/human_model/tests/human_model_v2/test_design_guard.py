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
"""Unimplemented v2 must never silently execute the v1 plant."""
import pytest
from protomotions.robot_configs.human_model.registry import get_model
from protomotions.robot_configs.human_model.profile import load_profile
from protomotions.robot_configs.smpl import SmplRobotConfig


def test_v2_can_be_inspected_but_not_executed(monkeypatch):
    monkeypatch.delenv("PROTOMOTIONS_HUMAN_MODEL", raising=False)
    model = get_model("human_model_v2", require_runnable=False)
    assert model.status == "candidate" and not model.runnable
    for operation in (lambda: get_model("human_model_v2"),
                      lambda: load_profile("human_model_v2"),
                      lambda: SmplRobotConfig(human_model_profile="human_model_v2"),
                      lambda: model.asset_path("usd")):
        with pytest.raises(NotImplementedError, match="design-only"):
            operation()


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="Unknown human model"):
        get_model("human_model_v20")
