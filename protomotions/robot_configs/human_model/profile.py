# SPDX-License-Identifier: Apache-2.0
"""Load and validate model data without importing a physics engine."""

import json
import math
from pathlib import Path


def load_profile(name="healthy_adult_v1"):
    # Only packaged, versioned profiles are accepted, including for old checkpoints.
    if not isinstance(name, str) or not name.replace("_", "").isalnum():
        raise ValueError(f"Invalid human model profile: {name!r}")
    path = Path(__file__).with_name(f"{name}.json")
    if not path.is_file():
        raise ValueError(f"Unknown human model profile: {name!r}")
    profile = json.loads(path.read_text())
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
    return profile
