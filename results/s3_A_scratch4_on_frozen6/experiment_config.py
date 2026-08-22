"""Curriculum v1 Stage 3: frozen ⑥ + trainable ④ walking imitation.

Same locomotion task, rewards, AMP setup and steering commands as
``mlp_locomotion_omni_uniform_speed_s80_task70_foot05`` (V11), so a ④ trained
here is directly comparable to V11 itself.  The only difference is that a frozen
⑥ sits underneath and adds a bounded posture correction to every action.

The point is the division of labour.  V11 learned balance on its own because it
had to; a ④ trained on top of a stabilizer can offload it.  Whether it actually
does is measured afterwards by sweeping ``HC_POSTURE_GAIN`` down to 0 -- if gait
degrades as ⑥ is weakened, ④ delegated; if nothing happens, it did not, and the
"④ = voluntary movement, ⑥ = balance" split is not real.

Pushes default OFF here.  V11 was trained with ``domain_randomization: None``,
and keeping the training condition identical is what makes the comparison
against V11 mean anything.  Disturbance belongs in the evaluation grid
(``HC_STEERING_PUSH``), not smuggled into training.

Knobs:
  HC_POSTURE_FOUNDATION_CKPT  frozen ⑥ (required)
  HC_MOTION_DRIVER_CKPT       optional ④ warm start; unset = ④ from scratch
  HC_POSTURE_GAIN             ⑥ authority; 1.0 for training, swept at eval
  HC_COMPOSE_PUSH_STAGE       0 (default) matches V11; 1-2 add disturbance
  HC_FREE_BALANCE_OBS / HC_FREE_FORCE_OBS   must match the frozen ⑥ checkpoint
"""

import importlib.util
import os
from pathlib import Path

from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import MdpComponent
from protomotions.simulator.base_simulator.config import (
    DomainRandomizationConfig,
    PushDomainRandomizationConfig,
)

from human_controller.posture_free import (
    BALANCE_OBS_DIM,
    FORCE_OBS_DIM,
    compute_free_posture_observation,
)

_BASE_PATH = Path(__file__).with_name(
    "mlp_locomotion_omni_uniform_speed_s80_task70_foot05.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "locomotion_omni_uniform_speed_s80_task70_foot05", _BASE_PATH
)
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

terrain_config = _BASE.terrain_config
scene_lib_config = _BASE.scene_lib_config
motion_lib_config = _BASE.motion_lib_config
apply_inference_overrides = _BASE.apply_inference_overrides

_PUSH_SETTINGS = {
    0: ((100000.0, 100001.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    1: ((2.0, 3.5), (0.20, 0.20, 0.0), (0.15, 0.15, 0.08)),
    2: ((1.25, 2.5), (0.35, 0.35, 0.0), (0.30, 0.30, 0.15)),
}

# Must match the frozen ⑥: balance block on, force off -> 14 + 13 + 3*69 = 234.
_BALANCE_OBS = os.environ.get("HC_FREE_BALANCE_OBS", "1") == "1"
_FORCE_OBS = os.environ.get("HC_FREE_FORCE_OBS", "0") == "1"


def configure_robot_and_simulator(robot_cfg, simulator_cfg, args):
    # Optional hook in ProtoMotions; the locomotion chain never defines one.
    base_fn = getattr(_BASE, "configure_robot_and_simulator", None)
    if base_fn is not None:
        base_fn(robot_cfg, simulator_cfg, args)
    stage = int(os.environ.get("HC_COMPOSE_PUSH_STAGE", "0"))
    if stage not in _PUSH_SETTINGS:
        raise ValueError("HC_COMPOSE_PUSH_STAGE must be 0, 1, or 2")
    if stage == 0:
        return
    interval, linear, angular = _PUSH_SETTINGS[stage]
    simulator_cfg.domain_randomization = DomainRandomizationConfig(
        push=PushDomainRandomizationConfig(
            push_interval_range=interval,
            max_linear_velocity=linear,
            max_angular_velocity=angular,
        )
    )


def env_config(robot_cfg, args):
    cfg = _BASE.env_config(robot_cfg, args)
    cfg.observation_components["posture_free_obs"] = MdpComponent(
        compute_func=compute_free_posture_observation,
        dynamic_vars={
            "root_rot": EnvContext.current.root_rot,
            "root_vel": EnvContext.current.root_vel,
            "root_ang_vel": EnvContext.current.root_ang_vel,
            "root_pos": EnvContext.current.root_pos,
            "ground_height": EnvContext.ground_heights,
            "rigid_body_contacts": EnvContext.current.rigid_body_contacts,
            "rigid_body_pos": EnvContext.current.rigid_body_pos,
            "contact_force_magnitudes": EnvContext.current_contact_force_magnitudes,
            "dof_pos": EnvContext.current.dof_pos,
            "dof_vel": EnvContext.current.dof_vel,
            "previous_processed_action": EnvContext.previous_processed_action,
        },
        static_params={
            "include_balance": _BALANCE_OBS,
            "include_force": _FORCE_OBS,
        },
    )
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    cfg = _BASE.agent_config(robot_cfg, env_cfg, args)
    if "posture_free_obs" not in cfg.model.in_keys:
        cfg.model.in_keys.append("posture_free_obs")
    if "posture_free_obs" not in cfg.model.actor.in_keys:
        cfg.model.actor.in_keys.append("posture_free_obs")
    # mu_model.in_keys stays ④'s; the wrapper reads posture_free_obs separately.
    cfg.model.actor.mu_model._target_ = (
        "human_controller.models.posture_stabilized_motion_driver."
        "PostureGroundedMotionDriver"
    )
    num_dof = int(robot_cfg.default_dof_pos.numel())
    obs_dim = (
        14
        + (BALANCE_OBS_DIM if _BALANCE_OBS else 0)
        + (FORCE_OBS_DIM if _FORCE_OBS else 0)
        + 3 * num_dof
    )
    os.environ["HC_FREE_OBS_DIM"] = str(obs_dim)
    try:
        cfg.model.actor.mu_model.posture_obs_dim = obs_dim
    except Exception:
        pass
    return cfg
