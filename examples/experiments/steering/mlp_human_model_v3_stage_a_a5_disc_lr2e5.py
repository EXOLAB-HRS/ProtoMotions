"""A5 with the discriminator learning rate lowered to 2e-5.

Single-variable screening for the A5 plateau: plant, motion pack, task/AMP
credit split, velocity-error scale, command dwell, PPO settings and network
stay exactly as in A5.  Only ``discriminator_optimizer.lr`` changes from 5e-5
to 2e-5.  Runtime env/batch, seed, transition budget and the motion pack remain
launcher inputs and must match the A5 run for the comparison to hold.
"""

import importlib.util
from pathlib import Path


_SPEC = importlib.util.spec_from_file_location(
    "a5_disc_lr_base", Path(__file__).with_name("mlp_human_model_v3_stage_a_a5.py")
)
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

terrain_config = _BASE.terrain_config
scene_lib_config = _BASE.scene_lib_config
motion_lib_config = _BASE.motion_lib_config
env_config = _BASE.env_config
apply_inference_overrides = _BASE.apply_inference_overrides
configure_robot_and_simulator = _BASE.configure_robot_and_simulator


def agent_config(robot_cfg, env_cfg, args):
    cfg = _BASE.agent_config(robot_cfg, env_cfg, args)
    cfg.model.discriminator_optimizer.lr = 2e-5
    return cfg
