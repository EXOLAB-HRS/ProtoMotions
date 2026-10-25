"""teacher_se_v1 stage 3 (2026-10-25): continues from se-v1-s2 last.ckpt with two changes.

- steady_speed_tracking (weight STEADY_SPEED_WEIGHT): the speed_transition_tracking error, scored only outside
  transitions (score_in="steady"). s2 walked ~0.17 m/s over the command at 0.8 / 1.0 / 1.4 m/s; in steady state
  speed was scored only by heading_rew (vel_err_scale 20), where 0.17 m/s keeps exp(-20 * 0.17^2) = 0.56 of the term.
- cadence_min_speed 0.5 -> CADENCE_MIN_SPEED: the cadence ratio command, the cadence terms and the Lencioni leg prior
  also act on slow walking (commands are 0.2-1.4 m/s already), for slow, short-step walking on command.
Everything else is s2 (r25 rewards, SE_v1 inputs, reset_speed_from_spawn).
"""
import copy

from examples.experiments.steering.mlp_human_model_v3_teacher_se_v1_s2 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_teacher_se_v1_s2 import env_config as _s2_env_config

STEADY_SPEED_WEIGHT = 0.15
CADENCE_MIN_SPEED = 0.3


def env_config(robot_cfg, args):
    cfg = _s2_env_config(robot_cfg, args)
    steady = copy.deepcopy(cfg.reward_components["speed_transition_tracking"])
    steady.static_params["score_in"] = "steady"
    steady.static_params["weight"] = STEADY_SPEED_WEIGHT
    cfg.reward_components["steady_speed_tracking"] = steady
    cfg.control_components["steering"].cadence_min_speed = CADENCE_MIN_SPEED
    return cfg
