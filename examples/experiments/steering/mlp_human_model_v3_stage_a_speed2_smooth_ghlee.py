"""a8 candidate: a7 plus a weak processed-action delta penalty."""
import importlib.util
from pathlib import Path

from protomotions.envs.component_factories import action_smoothness_factory

_spec = importlib.util.spec_from_file_location(
    "v3_speed2_smooth_base",
    Path(__file__).with_name("mlp_human_model_v3_stage_a_speed2_ghlee.py"),
)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

terrain_config = _base.terrain_config
scene_lib_config = _base.scene_lib_config
motion_lib_config = _base.motion_lib_config
agent_config = _base.agent_config
apply_inference_overrides = _base.apply_inference_overrides
configure_robot_and_simulator = _base.configure_robot_and_simulator


def env_config(robot_cfg, args):
    cfg = _base.env_config(robot_cfg, args)
    cfg.reward_components["action_smoothness"] = action_smoothness_factory(weight=-0.01)
    return cfg
