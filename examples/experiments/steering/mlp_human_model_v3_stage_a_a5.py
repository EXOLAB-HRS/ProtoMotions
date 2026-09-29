"""A5: local Stage A speed-tracking candidate on human_model_v3.

Relative to the published A3 configuration this candidate changes the
velocity-error scale, task/AMP credit split, discriminator learning rate, and
command dwell.  Runtime env/batch, seed, transition budget, and the exact
motion pack remain launcher inputs and must be recorded by the owning run.
"""

import importlib.util
from pathlib import Path


_SPEC = importlib.util.spec_from_file_location(
    "a5_base", Path(__file__).with_name("mlp_human_model_v3_stage_a.py")
)
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

terrain_config = _BASE.terrain_config
scene_lib_config = _BASE.scene_lib_config
motion_lib_config = _BASE.motion_lib_config
apply_inference_overrides = _BASE.apply_inference_overrides
configure_robot_and_simulator = _BASE.configure_robot_and_simulator


def env_config(robot_cfg, args):
    cfg = _BASE.env_config(robot_cfg, args)
    cfg.reward_components["heading_rew"].static_params["vel_err_scale"] = 2.0
    steering = cfg.control_components["steering"]
    steering.heading_change_steps_min = 300
    steering.heading_change_steps_max = 301
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    cfg = _BASE.agent_config(robot_cfg, env_cfg, args)
    cfg.task_reward_w = 0.7
    cfg.amp_parameters.discriminator_reward_w = 0.3
    cfg.model.discriminator_optimizer.lr = 5e-5
    cfg.save_last_checkpoint_every = 5
    cfg.save_epoch_checkpoint_every = 250
    return cfg
