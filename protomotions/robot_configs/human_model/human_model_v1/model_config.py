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
"""Current 69-coordinate model; opt-in candidates are not defaults."""
MODEL = {
    "model_id": "human_model_v1",
    "profile_id": "healthy_adult_v1",
    "status": "active",
    "runnable": True,
    "profile": "profiles/healthy_adult_v1.json",
    "assets": {"usd": "assets/smpl_humanoid_healthy_adult_v1.usda",
               "mjcf": "assets/smpl_humanoid.xml"},
    "joint_mode": "serial",
    "joint_frame_mass": 1e-6,
    "default_features": [],
    "candidate_profiles": {"population_profile_candidate": "profiles/population_profile_candidate.json"},
}
