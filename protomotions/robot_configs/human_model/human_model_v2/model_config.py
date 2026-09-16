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
"""Authorized design namespace, not an executable anatomical model."""
MODEL = {
    "model_id": "human_model_v2",
    "profile_id": None,
    "status": "candidate",
    "runnable": False,
    "reason": "Anatomical assets, coordinates, policy adapters and validation are not implemented.",
    "planned_lower_limb_dofs_per_side": {"hip": 3, "knee": 1, "ankle": 1, "subtalar": 1, "mtp": 1},
    "knee_coupled_translation": False,
}
