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
"""Stable package resources, independent of legacy import wrappers."""
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROTOMOTIONS_ROOT = PACKAGE_ROOT.parents[2]
WORKSPACE_ROOT = PROTOMOTIONS_ROOT.parent
ASSET_ROOT = PACKAGE_ROOT.parents[1] / "data/assets"
V1_PROFILE_ROOT = PACKAGE_ROOT / "human_model_v1/profiles"

def v1_resource(name):
    return V1_PROFILE_ROOT / name
