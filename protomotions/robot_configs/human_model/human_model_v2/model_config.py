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
"""Executable v2 plant; teacher adapters deliberately outside this stage."""
MODEL = {
    "model_id": "human_model_v2", "profile_id": "human_model_v2",
    "status": "candidate", "runnable": True,
    "assets": {"usd": "assets/human_model_v2.usda", "mjcf": "assets/human_model_v2.xml"},
    "profile": "profiles/healthy_adult_v2.json",
    "lower_limb_dofs_per_side": {"hip": 3, "knee": 1, "ankle": 1, "subtalar": 1, "mtp": 1},
    "knee_coupled_translation": False,
}


def robot_config():
    from protomotions.robot_configs.smpl import SmplRobotConfig
    from protomotions.robot_configs.base import RobotAssetConfig, ControlConfig, ControlType
    from protomotions.components.pose_lib import ControlInfo
    from pathlib import Path
    root=Path(__file__).parent
    # The regular SMPL config machinery supports 1/3-DOF joints and arbitrary axes.
    return SmplRobotConfig(human_model_profile='human_model_v2',human_model_usd_joint_mode='anatomical',
        control=ControlConfig(control_type=ControlType.TORQUE,override_control_info={'.*':ControlInfo(stiffness=500.,damping=50.,effort_limit=500.,velocity_limit=100.)}),
        asset=RobotAssetConfig(asset_root=str(root/'assets'),asset_file_name='human_model_v2.xml',
            usd_asset_file_name='human_model_v2.usda',usd_bodies_root_prim_path='/World/envs/env_.*/Robot/bodies/',
            self_collisions=False,max_linear_velocity=1000.,max_angular_velocity=1000.,angular_damping=0.,linear_damping=0.))


def configure_trunk_candidate(robot_cfg, simulator_cfg):
    """Opt-in e05 trunk candidate; defaults and existing runs stay unchanged.

    Call after the steering experiment's normal configuration. Coefficients
    are embedded in the robot config, so saved configurations are self-contained.
    The 480 Hz requirement belongs to this explicit-force numerical candidate;
    it is not a biological parameter. See the source/mapping limits in JSON.
    """
    import json
    from pathlib import Path
    from protomotions.robot_configs.base import ControlType
    if robot_cfg.human_model_profile!='human_model_v2':
        raise ValueError('The trunk candidate requires human_model_v2')
    if '.isaaclab.' not in simulator_cfg._target_:
        raise ValueError('The trunk PD candidate was assessed on Isaac Lab only')
    if robot_cfg.control.control_type!=ControlType.PROPORTIONAL:
        raise ValueError('Configure the steering q-target PD interface first')
    candidate=json.loads((Path(__file__).parent/'profiles/trunk_candidate.json').read_text())
    if getattr(robot_cfg,'human_model_parameters',{}):
        raise ValueError('Do not silently combine trunk candidate with another plant candidate')
    robot_cfg.human_model_parameters=candidate['human_model_parameters']
    pd=candidate['pd_candidate']
    for body in ('Torso','Spine','Chest'):
        for axis in 'xyz':
            control=robot_cfg.control.control_info[f'{body}_{axis}']
            # Do not depend on the caller's previous trunk gains.
            base=control.effort_limit/1.5
            control.stiffness=base*pd['trunk_kp_multiplier']
            control.damping=base*.1*pd['trunk_kd_multiplier']
    simulator_cfg.sim.fps=pd['physics_fps']
    simulator_cfg.sim.decimation=pd['physics_fps']//pd['control_fps']
