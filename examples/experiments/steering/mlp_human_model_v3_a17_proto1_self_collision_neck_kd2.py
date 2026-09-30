"""a17: scratch ProtoMotions steering with self-collision and doubled neck Kd."""

from examples.experiments.steering import mlp_human_model_v3_a11_proto1 as _BASE

EXPERIMENT_ID = "a17"
NECK_KD_MULTIPLIER = 2.0

terrain_config = _BASE.terrain_config
scene_lib_config = _BASE.scene_lib_config
motion_lib_config = _BASE.motion_lib_config
env_config = _BASE.env_config
agent_config = _BASE.agent_config
apply_inference_overrides = _BASE.apply_inference_overrides


def configure_robot_and_simulator(robot_cfg, simulator_cfg, args):
    """Keep the a11 one-shot recipe, changing only collision and neck damping."""
    _BASE.configure_robot_and_simulator(robot_cfg, simulator_cfg, args)
    robot_cfg.asset.self_collisions = True
    neck_axes = [name for name in robot_cfg.control.control_info if name.startswith("Neck_")]
    if set(neck_axes) != {"Neck_x", "Neck_y", "Neck_z"}:
        raise ValueError(f"Expected Neck_x/y/z PD gains, found {neck_axes}")
    for name in neck_axes:
        robot_cfg.control.control_info[name].damping *= NECK_KD_MULTIPLIER
