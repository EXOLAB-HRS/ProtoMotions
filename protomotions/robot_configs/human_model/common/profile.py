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
"""Load and validate model data without importing a physics engine."""

import json
import math
from ..registry import profile_path


def load_profile(name="healthy_adult_v1"):
    # Only packaged, versioned profiles are accepted, including for old checkpoints.
    if not isinstance(name, str) or not name.replace("_", "").isalnum():
        raise ValueError(f"Invalid human model profile: {name!r}")
    path = profile_path(name)
    if not path.is_file():
        raise ValueError(f"Unknown human model profile: {name!r}")
    profile = json.loads(path.read_text())
    return validate_profile(profile)


def validate_profile(profile):
    """Validate both packaged data and explicitly supplied in-memory profiles."""
    if profile["schema_version"] != 1:
        raise ValueError("Unsupported human model profile schema")
    for name, joint in profile["joints"].items():
        lo, hi = joint["rom_deg"]
        if not all(math.isfinite(x) for x in (lo, hi)) or not lo < hi:
            raise ValueError(f"Invalid ROM: {name}")
        for field in ("active_nm", "stiffness_nm_per_rad"):
            if len(joint[field]) != 2 or any(
                not math.isfinite(x) or x < 0 for x in joint[field]
            ):
                raise ValueError(f"Invalid {field}: {name}")
        if not math.isfinite(joint["damping_nms_per_rad"]) or joint["damping_nms_per_rad"] < 0:
            raise ValueError(f"Invalid damping: {name}")
        if not lo <= joint["rest_deg"] <= hi:
            raise ValueError(f"Rest position outside ROM: {name}")
        for source in joint["evidence"].values():
            if source not in profile["sources"]:
                raise ValueError(f"Unknown evidence reference: {source}")
    pose_strength = profile.get("pose_strength_model")
    if pose_strength is not None:
        validate_pose_strength_model(pose_strength, profile)
    return profile


def validate_pose_strength_model(pose_strength, profile):
    """Validate a serialized, opt-in table of posture-dependent torque caps."""
    if set(pose_strength) != {"status", "sources", "curves"} or not isinstance(pose_strength["status"], str):
        raise ValueError("pose_strength_model requires status, sources, and curves")
    if not isinstance(pose_strength["sources"], dict):
        raise ValueError("pose_strength_model sources must be a mapping")
    known_sources = set(profile["sources"]) | set(pose_strength["sources"])
    seen = set()
    for curve in pose_strength["curves"]:
        required = {"joint", "conditioning_joints", "conditioning_weights", "direction", "angles_rad", "torques_nm", "source"}
        if set(curve) != required:
            raise ValueError("Invalid pose strength curve fields")
        joint = curve["joint"]
        conditioning = curve["conditioning_joints"]
        weights = curve["conditioning_weights"]
        direction = curve["direction"]
        if (joint not in profile["joints"] or not conditioning
                or any(name not in profile["joints"] for name in conditioning)):
            raise ValueError(f"Unknown pose strength joint: {joint}/{conditioning}")
        if len(conditioning) != len(weights) or any(not math.isfinite(x) for x in weights):
            raise ValueError(f"Invalid pose strength conditioning: {joint}")
        if direction not in ("negative", "positive") or (joint, direction) in seen:
            raise ValueError(f"Duplicate or invalid pose strength direction: {joint}/{direction}")
        seen.add((joint, direction))
        angles = curve["angles_rad"]
        torques = curve["torques_nm"]
        if (len(angles) < 2 or len(angles) != len(torques)
                or any(not math.isfinite(x) for x in (*angles, *torques))
                or any(x < 0 for x in torques)
                or any(b <= a for a, b in zip(angles, angles[1:]))):
            raise ValueError(f"Invalid pose strength samples: {joint}/{direction}")
        if curve["source"] not in known_sources:
            raise ValueError(f"Unknown pose strength source: {curve['source']}")
    return pose_strength
