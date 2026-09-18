"""Opt-in trunk candidate teacher; owned by e06_trunk_teacher_ghlee.

Uses the existing v3 references and Stage A task. Trunk passive damping remains
an engineering prior; the e05 harness assay does not certify autonomous gait.
"""
import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "stage_a_trunk_ghlee_base", Path(__file__).with_name("mlp_human_model_v2_stage_a_forward.py")
)
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

terrain_config = _BASE.terrain_config
scene_lib_config = _BASE.scene_lib_config
env_config = _BASE.env_config
apply_inference_overrides = _BASE.apply_inference_overrides


def configure_robot_and_simulator(robot_cfg, simulator_cfg, args):
    from protomotions.robot_configs.human_model.human_model_v2.model_config import configure_trunk_candidate
    _BASE.configure_robot_and_simulator(robot_cfg, simulator_cfg, args)
    configure_trunk_candidate(robot_cfg, simulator_cfg)


def motion_lib_config(args):
    cfg = _BASE.motion_lib_config(args)
    cfg._target_ = "protomotions.robot_configs.human_model.human_model_v2.motion_lib.HumanModelMotionLib"
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    cfg = _BASE.agent_config(robot_cfg, env_cfg, args)
    cfg.task_reward_w = 0.6
    cfg.amp_parameters.discriminator_reward_w = 0.4
    cfg.save_last_checkpoint_every = 5
    cfg.save_epoch_checkpoint_every = 250
    return cfg
