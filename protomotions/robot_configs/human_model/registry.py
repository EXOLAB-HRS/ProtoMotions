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
"""Select model resources without silently running v1 under a v2 label."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from .human_model_v1.model_config import MODEL as V1
from .human_model_v2.model_config import MODEL as V2

_PACKAGE = Path(__file__).resolve().parent

@dataclass(frozen=True)
class ModelDefinition:
    model_id: str
    profile_id: str | None
    status: str
    runnable: bool

    @property
    def root(self):
        return _PACKAGE / self.model_id

    def asset_path(self, kind):
        if not self.runnable:
            raise NotImplementedError(f"{self.model_id} is design-only; no executable assets")
        return self.root / V1["assets"][kind]


def get_model(name="human_model_v1", *, require_runnable=True):
    if name in ("human_model_v1", "healthy_adult_v1"):
        data = V1
    elif name == "human_model_v2":
        data = V2
    else:
        raise ValueError(f"Unknown human model: {name!r}")
    model = ModelDefinition(data["model_id"], data["profile_id"], data["status"], data["runnable"])
    if require_runnable and not model.runnable:
        raise NotImplementedError(f"{model.model_id} is design-only: {data['reason']}")
    return model


def profile_path(name="healthy_adult_v1"):
    model = get_model(name)
    return model.root / V1["profile"]
