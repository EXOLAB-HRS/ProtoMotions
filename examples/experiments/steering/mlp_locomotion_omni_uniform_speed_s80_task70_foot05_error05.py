"""④+⑤ MVP: frozen V11 Motion Driver plus a bounded temporal Error Corrector.

The environment, AMP data, reward, command sampling, critic, and discriminator
are inherited from V11.  Only the actor mean model is replaced by the ④+⑤
composition.  ④ is restored and frozen from ``HC_MOTION_DRIVER_CKPT``; ⑤ starts
at exactly zero and has a maximum action residual of 0.05.
"""

import importlib.util
import os
from pathlib import Path

_BASE_PATH = Path(__file__).with_name("mlp_locomotion_omni_uniform_speed_s80_task70_foot05.py")
_SPEC = importlib.util.spec_from_file_location("locomotion_v11_base", _BASE_PATH)
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

terrain_config = _BASE.terrain_config
scene_lib_config = _BASE.scene_lib_config
motion_lib_config = _BASE.motion_lib_config
env_config = _BASE.env_config
apply_inference_overrides = _BASE.apply_inference_overrides


def env_config(robot_cfg, args):
    """Add only the delayed action observation needed for ⑤ efference copy.

    The frozen V11 actor configuration remains unchanged: ④ never consumes
    this key.  It is visible only to the ⑤ wrapper.
    """
    from protomotions.envs.component_factories import previous_actions_factory

    cfg = _BASE.env_config(robot_cfg, args)
    cfg.observation_components["previous_actions"] = previous_actions_factory(history_steps=1)
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    cfg = _BASE.agent_config(robot_cfg, env_cfg, args)
    cfg.model.actor.mu_model._target_ = "human_controller.models.error_corrected_motion_driver.ErrorCorrectedMotionDriver"
    # ④ is frozen.  This LR therefore controls only the ⑤ forward model and
    # correction head; the identification phase needs a materially larger LR
    # than the conservative residual-PPO smoke.
    cfg.model.actor_optimizer.lr = float(os.environ.get("HC_ERROR_ACTOR_LR", "1e-4"))
    return cfg
