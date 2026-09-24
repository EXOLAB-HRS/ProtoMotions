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
"""Apply a profile at simulation creation, including unpickled old checkpoints."""

from copy import deepcopy
import logging
import hashlib
import os
from pathlib import Path

from ..human_model_v1.assets.builder import derive_mjcf, derive_serial_usda, validate_usda_rom
from .dynamics import HumanJointModel
from .profile import load_profile
from .paths import ASSET_ROOT
from ..registry import get_model

log = logging.getLogger(__name__)


def selected_profile(robot_config):
    if type(robot_config).__name__ != "SmplRobotConfig":
        return None
    name = os.environ.get("PROTOMOTIONS_HUMAN_MODEL", getattr(robot_config, "human_model_profile", "healthy_adult_v1"))
    return None if name in (None, "legacy") else name


def selected_model_parameters(robot_config):
    """Explicit serialized plant parameters; old checkpoints default to empty.

    The existing FEATURES environment override takes precedence when present.
    No experiment file is discovered or silently promoted into runtime defaults.
    """
    parameters = deepcopy(getattr(robot_config,'human_model_parameters',{}))
    allowed = {'features','strength_cohort','strength_reference_size','active_strength_scale','fatigue_regions','fatigue_rest_multiplier','passive_joint_parameters','pose_strength_model'}
    if not isinstance(parameters,dict) or set(parameters)-allowed:
        raise ValueError('human_model_parameters contains unsupported parameters')
    if 'PROTOMOTIONS_HUMAN_MODEL_FEATURES' in os.environ:
        parameters['features'] = tuple(filter(None,(s.strip() for s in os.environ['PROTOMOTIONS_HUMAN_MODEL_FEATURES'].split(','))))
    if isinstance(parameters.get('features',()),str):
        raise ValueError('human_model_parameters.features must be a sequence of feature names')
    return parameters


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
    model = HumanJointModel(load_profile(name), info.dof_names, info.dof_limits_lower.device,
                            **selected_model_parameters(robot_config))
    info.dof_limits_lower = model.lower
    info.dof_limits_upper = model.upper
    for i, dof in enumerate(info.dof_names):
        robot_config.control.control_info[dof].effort_limit = max(float(model.negative[i]), float(model.positive[i]))
    return model


def prepare_simulator(robot_config, simulator_config, device):
    """Return (backend config, force model), leaving command type and PD gains intact.

    Update shared kinematic metadata (old checkpoints skip __post_init__), then
    copy only for backend force ceilings and model asset paths. Policy
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
    model = HumanJointModel(profile, backend.kinematic_info.dof_names, device,
                            **selected_model_parameters(robot_config))
    if "activation" in model.features:
        import torch
        # Allocate before the first snapshot/reset; no hidden lazy state appears
        # after capture of a fresh environment.
        model._activation = torch.zeros((simulator_config.num_envs, len(model.names), 2),
                                        device=device, dtype=model.lower.dtype)
    if 'fatigue' in model.features:
        import torch
        model._fatigue = torch.zeros((simulator_config.num_envs,len(model.names),2,3),
                                     device=device,dtype=model.lower.dtype)
        model._fatigue[...,0] = 1
    limits = {n: float(model.backend_limit[i]) for i, n in enumerate(model.names)}
    for n, limit in limits.items():
        backend.control.control_info[n].effort_limit = limit
        backend.control.control_info[n].friction = 0.0
    asset = backend.asset
    packaged_assets = ASSET_ROOT
    if ".isaaclab." in target:
        # Load the committed, editable USDA directly. RoM is authored in the
        # file; actuator force settings are supplied by the IsaacLab scene.
        usd = get_model(name).asset_path("usd")
        if name in ("human_model_v2", "human_model_v3"):
            from ..human_model_v2.assets.builder import validate_assets
            validate_assets(profile)
        else:validate_usda_rom(usd.read_text(), profile)
        joint_mode=getattr(robot_config,'human_model_usd_joint_mode','serial')
        backend.human_model_usd_joint_mode=joint_mode
        if joint_mode=='serial':
            usd=derive_serial_usda(usd,frame_mass=getattr(robot_config,'human_model_joint_frame_mass',1e-6))
        elif joint_mode not in ('d6','anatomical'):
            raise ValueError(f'Unsupported human model USD joint mode: {joint_mode}')
        collision_profile = getattr(robot_config, 'human_model_collision_profile', None)
        if collision_profile is not None:
            if collision_profile != 'human_model_v3.1' or name != 'human_model_v3' or joint_mode != 'anatomical':
                raise ValueError('human_model_v3.1 collision revision requires the anatomical v3 plant')
            from ..human_model_v3_1.collision import validate_collision_asset
            corrected = get_model(collision_profile).asset_path('usd')
            validate_collision_asset(usd, corrected)
            usd = corrected
        asset.asset_root = str(usd.parent)
        asset.usd_asset_file_name = usd.name
        backend._human_model_asset_sha256=hashlib.sha256(usd.read_bytes()).hexdigest()
    else:
        xml = Path(asset.asset_root) / asset.asset_file_name
        if not xml.is_file():
            # Checkpoint may contain an absolute path on its training machine.
            xml = packaged_assets / "mjcf/smpl_humanoid.xml"
        if name in ("human_model_v2", "human_model_v3"):
            raise NotImplementedError("v2 runtime currently verified on IsaacLab only")
        asset.asset_root = str(derive_mjcf(xml, profile, limits))
        asset.asset_file_name = "smpl_humanoid.xml"
    backend._human_model_enabled = True
    if ".mujoco." in target:
        simulator_config.use_implicit_pd = False
    log.info("Human model %s: physiological ROM, direction-specific active caps, separate passive forces; assets %s", name, asset.asset_root)
    return backend, model
