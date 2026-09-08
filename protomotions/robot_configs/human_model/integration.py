# SPDX-License-Identifier: Apache-2.0
"""Apply a profile at simulation creation, including unpickled old checkpoints."""

from copy import deepcopy
import logging
import os
from pathlib import Path

from .assets import derive_assets
from .dynamics import HumanJointModel
from .profile import load_profile

log = logging.getLogger(__name__)


def selected_profile(robot_config):
    if type(robot_config).__name__ != "SmplRobotConfig":
        return None
    name = os.environ.get("PROTOMOTIONS_HUMAN_MODEL", getattr(robot_config, "human_model_profile", "healthy_adult_v1"))
    return None if name in (None, "legacy") else name


def apply_metadata(robot_config):
    """New configs expose physiological limits to rewards and action processors."""
    name = selected_profile(robot_config)
    if name is None:
        return None
    info = robot_config.kinematic_info
    if not hasattr(robot_config, "_human_model_original_limits"):
        robot_config._human_model_original_limits = (
            info.dof_limits_lower.clone(), info.dof_limits_upper.clone(),
            {n: c.effort_limit for n, c in robot_config.control.control_info.items()},
        )
    model = HumanJointModel(load_profile(name), info.dof_names, info.dof_limits_lower.device)
    info.dof_limits_lower = model.lower
    info.dof_limits_upper = model.upper
    for i, dof in enumerate(info.dof_names):
        robot_config.control.control_info[dof].effort_limit = max(float(model.negative[i]), float(model.positive[i]))
    return model


def prepare_simulator(robot_config, simulator_config, device):
    """Return (backend config, force model), leaving command type and PD gains intact.

    Update shared kinematic metadata (old checkpoints skip __post_init__), then
    copy only for backend force ceilings and generated asset paths. Policy
    parameters, action ordering, controller gains and source assets are preserved.
    """
    name = selected_profile(robot_config)
    if name is None:
        if hasattr(robot_config, "_human_model_original_limits"):
            lower, upper, effort = robot_config._human_model_original_limits
            robot_config.kinematic_info.dof_limits_lower = lower.clone()
            robot_config.kinematic_info.dof_limits_upper = upper.clone()
            for n, limit in effort.items():
                robot_config.control.control_info[n].effort_limit = limit
        return robot_config, None
    target = simulator_config._target_
    # Fail visibly instead of silently dropping physiological forces on an
    # unreviewed engine. Additional engines can be enabled with adapter tests.
    if not any(f".{engine}." in target for engine in ("mujoco", "isaaclab", "isaacgym")):
        raise ValueError(f"Human model adapter not validated for {target}; use a supported engine or PROTOMOTIONS_HUMAN_MODEL=legacy")
    profile = load_profile(name)
    apply_metadata(robot_config)
    backend = deepcopy(robot_config)
    model = HumanJointModel(profile, backend.kinematic_info.dof_names, device)
    limits = {n: float(model.backend_limit[i]) for i, n in enumerate(model.names)}
    for n, limit in limits.items():
        backend.control.control_info[n].effort_limit = limit
        backend.control.control_info[n].friction = 0.0
    asset = backend.asset
    root = Path(asset.asset_root)
    xml = root / asset.asset_file_name
    usd = root / asset.usd_asset_file_name
    if not xml.is_file():
        # Checkpoint may contain an absolute path on its training machine.
        root = Path(__file__).resolve().parents[2] / "data/assets"
        xml, usd = root / "mjcf/smpl_humanoid.xml", root / "usd/smpl_humanoid.usda"
    generated = derive_assets(xml, usd, profile, limits, require_usd=".isaaclab." in target)
    asset.asset_root = str(generated)
    asset.asset_file_name = "smpl_humanoid.xml"
    asset.usd_asset_file_name = "smpl_humanoid.usda"
    backend._human_model_enabled = True
    if ".mujoco." in target:
        simulator_config.use_implicit_pd = False
    log.info("Human model %s: physiological ROM, direction-specific active caps, separate passive forces; assets %s", name, generated)
    return backend, model
