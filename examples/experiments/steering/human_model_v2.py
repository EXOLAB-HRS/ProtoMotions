"""Candidate steering teacher on fixed v2 plant; PPO+AMP with gait safeguards."""
import importlib.util
from pathlib import Path
import torch
import os
import hashlib
from protomotions.robot_configs.base import ControlType
from protomotions.envs.control.walking_quality import WalkingSteeringConfig, walking_penalty
from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import MdpComponent

_spec = importlib.util.spec_from_file_location('base_steering',Path(__file__).with_name('mlp.py'))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
terrain_config = _base.terrain_config
scene_lib_config = _base.scene_lib_config
def agent_config(robot_config, env_config, args):
    cfg = _base.agent_config(robot_config, env_config, args)
    if os.environ.get('HC_STEERING_SMOKE') == '1':
        if args.num_envs > 8 or args.training_max_steps > 1024:
            raise ValueError('Candidate smoke is limited to 8 environments and 1024 steps')
        cfg.batch_size = 256
        cfg.amp_parameters.discriminator_batch_size = 256
    return cfg
apply_inference_overrides = _base.apply_inference_overrides


def configure_robot_and_simulator(robot_cfg, simulator_cfg, args):
    if getattr(robot_cfg,'human_model_profile',None)!='human_model_v2' or args.simulator!='isaaclab':
        raise ValueError('This experiment requires --robot-name human_model_v2 --simulator isaaclab')
    # Explicitly interpret actions as q_ref; HumanJointModel applies PD and strength
    # clipping at each physics step, then adds passive torque exactly once.
    robot_cfg.control.control_type = ControlType.PROPORTIONAL
    simulator_cfg.sim.fps = 480
    simulator_cfg.sim.decimation = 16
    simulator_cfg.sim.physx.num_position_iterations = 32
    simulator_cfg.sim.physx.num_velocity_iterations = 1


def motion_lib_config(args):
    data = torch.load(args.motion_file,map_location='cpu',weights_only=False)
    meta = data.get('human_model_metadata',{})
    if meta.get('model_id')!='human_model_v2' or data['dps'].shape[-1]!=59 or data['gts'].shape[-2]!=26:
        raise ValueError('A retargeted v2 reference with provenance is required; raw SMPL motions are incompatible')
    from protomotions.robot_configs.human_model.common.paths import PACKAGE_ROOT
    from protomotions.components.pose_lib import extract_kinematic_info
    asset=PACKAGE_ROOT/'human_model_v2/assets/human_model_v2.xml'
    ki=extract_kinematic_info(str(asset))
    if (meta.get('dof_names')!=ki.dof_names or meta.get('body_names')!=ki.body_names
            or meta.get('asset_sha256')!=hashlib.sha256(asset.read_bytes()).hexdigest()):
        raise ValueError('Reference joint/body ordering or asset SHA does not match fixed v2')
    if meta.get('status')!='passed' and os.environ.get('HC_STEERING_SMOKE') != '1':
        raise ValueError('Reference motion quality gate has not passed; do not train on a failed retarget')
    if os.environ.get('HC_STEERING_SMOKE') == '1' and (args.num_envs>8 or args.training_max_steps>1024):
        raise ValueError('Candidate references only permitted for bounded smoke')
    cfg = _base.motion_lib_config(args)
    cfg._target_ = "protomotions.robot_configs.human_model.human_model_v2.motion_lib.HumanModelMotionLib"
    return cfg


def env_config(robot_cfg,args):
    if robot_cfg.control.control_type!=ControlType.PROPORTIONAL:
        raise ValueError('PD-to-human-torque adapter must be explicitly configured')
    cfg = _base.env_config(robot_cfg,args)
    cfg.control_components['steering'] = WalkingSteeringConfig(
        tar_speed_min=.5,tar_speed_max=1.5,random_speed_probability=1.,
        heading_change_steps_min=180,heading_change_steps_max=301,
        enable_rand_facing=False,stop_probability=0.)
    # Start with forward walking and heading changes. Strafe/backward/stop-start
    # require separate reference coverage and gates rather than forward cadence rules.
    cfg.reward_components['walking_quality'] = MdpComponent(
        compute_func=walking_penalty,
        dynamic_vars={'penalty':EnvContext.steering.walking_quality_penalty},
        static_params={'weight':-.15})
    return cfg
