"""a15: a13-B fine-tune with Isaac self-collision enabled."""
import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "a15_base", Path(__file__).with_name("mlp_human_model_v3_a13_b.py")
)
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)
terrain_config = _BASE.terrain_config
scene_lib_config = _BASE.scene_lib_config
motion_lib_config = _BASE.motion_lib_config
env_config = _BASE.env_config
agent_config = _BASE.agent_config
apply_inference_overrides = _BASE.apply_inference_overrides


def configure_robot_and_simulator(robot_cfg, simulator_cfg, args):
    _BASE.configure_robot_and_simulator(robot_cfg, simulator_cfg, args)
    robot_cfg.asset.self_collisions = True
