"""Candidate Stage A on v3. Kinematic v2 motion packs remain SHA-validated."""
import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('v3_stage_a_base',Path(__file__).with_name('mlp_human_model_v2_stage_a_trunk_ghlee.py'))
base=importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
terrain_config=base.terrain_config
scene_lib_config=base.scene_lib_config
motion_lib_config=base.motion_lib_config
env_config=base.env_config
agent_config=base.agent_config
apply_inference_overrides=base.apply_inference_overrides


def configure_robot_and_simulator(robot_cfg,simulator_cfg,args):
    from protomotions.robot_configs.human_model.human_model_v3.model_config import configure_pd
    if robot_cfg.human_model_profile!='human_model_v3':
        raise ValueError('Requires --robot-name human_model_v3')
    configure_pd(robot_cfg,simulator_cfg)
