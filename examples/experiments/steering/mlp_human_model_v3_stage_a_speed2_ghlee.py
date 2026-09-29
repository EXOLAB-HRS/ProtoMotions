"""a7 candidate: a4 Stage A setup with a stronger velocity-error gradient."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "v3_speed2_base", Path(__file__).with_name("mlp_human_model_v3_stage_a.py")
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
    cfg.reward_components["heading_rew"].static_params["vel_err_scale"] = 2.0
    return cfg
