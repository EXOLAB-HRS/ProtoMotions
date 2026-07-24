"""S1 configuration: frozen V11 ④ with trainable sensory-only ⑥."""
import os
import importlib.util
from pathlib import Path

_PATH = Path(__file__).with_name("mlp_posture_stand_push.py")
_SPEC = importlib.util.spec_from_file_location("posture_stand", _PATH)
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

terrain_config = _BASE.terrain_config
scene_lib_config = _BASE.scene_lib_config
motion_lib_config = _BASE.motion_lib_config
env_config = _BASE.env_config
configure_robot_and_simulator = _BASE.configure_robot_and_simulator
apply_inference_overrides = _BASE.apply_inference_overrides


def agent_config(robot_cfg, env_cfg, args):
    cfg = _BASE.agent_config(robot_cfg, env_cfg, args)
    cfg.model.in_keys.append("posture_obs")
    cfg.model.actor.in_keys.append("posture_obs")
    cfg.model.actor.mu_model._target_ = "human_controller.models.posture_stabilized_motion_driver.PostureStabilizedMotionDriver"
    cfg.model.actor_optimizer.lr = float(os.environ.get("HC_POSTURE_ACTOR_LR", "5e-5"))
    return cfg
