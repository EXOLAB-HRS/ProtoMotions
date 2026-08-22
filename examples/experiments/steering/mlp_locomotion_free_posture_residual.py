"""S2: frozen ④ (V11 locomotion) with the anchor-free ⑥ as a trainable residual.

The task is ④'s, unchanged -- same steering commands, same locomotion rewards,
same AMP setup as ``mlp_locomotion_omni_uniform_speed_s80_task70_foot05``.  The
only additions are ⑥'s observation bundle and a residual head on top of ④'s
action.  Keeping the task fixed is what makes the ⑥-disabled control
(``HC_POSTURE_ENABLED=0``) a real baseline: any difference is attributable to
the residual rather than to a changed objective.

⑥ is deliberately *not* asked to learn stepping here.  Stepping is ④'s job and
is already the main thing ④ was trained to do; ⑥'s job is to avoid destabilising
④ while it steps.  This is also why the support-margin reward is absent -- it
penalises the weight shift that recovery needs, which is how it failed when
tested on standalone ⑥.

Pushes are switched ON by default here, and that is load-bearing rather than a
detail.  ④ was trained with ``domain_randomization: None`` and already scores
``min_upright = 1.0`` on both the forward and directional sweeps, so on the
undisturbed task there is nothing left for ⑥ to contribute.  Combined with the
residual L2 penalty in ``run_train_posture_stabilizer.py`` the optimal ⑥ would be
exactly zero, and every arm would converge to "do nothing".  The disturbance is
what creates a gradient toward ⑥ being useful at all -- the same reason the
standalone ⑥ comparisons were meaningless until they were scored under push.

Knobs:
  HC_MOTION_DRIVER_CKPT       frozen ④ checkpoint (required)
  HC_POSTURE_FOUNDATION_CKPT  standalone ⑥ to warm-start from; unset = zero-init
  HC_POSTURE_ENABLED          0 disables the residual entirely (④-alone control)
  HC_POSTURE_RESIDUAL_SCALE   residual authority, default 0.40 (⑥'s trained value)
  HC_COMPOSE_PUSH_STAGE       0 none, 1 (default) matches ⑥'s training, 2 harder
  HC_FREE_BALANCE_OBS / HC_FREE_FORCE_OBS   must match the warm-start checkpoint
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

# Same magnitudes as the standalone ⑥ curriculum, so a warm-started ⑥ meets the
# disturbance distribution it was trained on and only the moving target is new.
_PUSH_SETTINGS = {
    0: ((100000.0, 100001.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    1: ((2.0, 3.5), (0.20, 0.20, 0.0), (0.15, 0.15, 0.08)),
    2: ((1.25, 2.5), (0.35, 0.35, 0.0), (0.30, 0.30, 0.15)),
}


def configure_robot_and_simulator(robot_cfg, simulator_cfg, args):
    # The hook is optional in ProtoMotions and the locomotion chain never defines
    # one -- ④ takes the factory defaults.  Guarded so this file keeps working if
    # a base in the chain later grows one.
    base_fn = getattr(_BASE, "configure_robot_and_simulator", None)
    if base_fn is not None:
        base_fn(robot_cfg, simulator_cfg, args)
    stage = int(os.environ.get("HC_COMPOSE_PUSH_STAGE", "1"))
    if stage not in _PUSH_SETTINGS:
        raise ValueError("HC_COMPOSE_PUSH_STAGE must be 0, 1, or 2")
    interval, linear, angular = _PUSH_SETTINGS[stage]
    simulator_cfg.domain_randomization = DomainRandomizationConfig(
        push=PushDomainRandomizationConfig(
            push_interval_range=interval,
            max_linear_velocity=linear,
            max_angular_velocity=angular,
        )
    )

# Defaults match the recommended ⑥ (p6_F_seed1_25m): balance block on, force off,
# giving 14 + 13 + 3*69 = 234.  A mismatch here makes the warm-start load fail
# loudly rather than silently training a differently-shaped net.
_BALANCE_OBS = os.environ.get("HC_FREE_BALANCE_OBS", "1") == "1"
_FORCE_OBS = os.environ.get("HC_FREE_FORCE_OBS", "0") == "1"


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
    # mu_model.in_keys stays ④'s: the wrapper feeds them to the frozen base and
    # reads posture_free_obs off the tensordict separately.
    cfg.model.actor.mu_model._target_ = (
        "human_controller.models.posture_stabilized_motion_driver."
        "FreePostureOnMotionDriver"
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
    cfg.model.actor_optimizer.lr = float(os.environ.get("HC_POSTURE_ACTOR_LR", "5e-5"))
    return cfg
