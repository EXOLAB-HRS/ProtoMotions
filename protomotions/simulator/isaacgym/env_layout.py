"""Isaac Gym actor placement independent of controller rollout state."""
from __future__ import annotations


def actor_start_offset(
    env_id: int, *, default_root_height: float
) -> tuple[float, float, float]:
    """Create every isolated actor at the same neutral local origin."""
    if env_id < 0:
        raise ValueError("env_id must be non-negative")
    return 0.0, 0.0, float(default_root_height)
