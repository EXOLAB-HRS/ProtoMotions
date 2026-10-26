"""teacher_se_v1 stage 3 with stopping emphasis (2026-10-25, training setting d stage 2 from scratch stage 1 s1c).

s3 (steady speed term 0.15, cadence floor 0.3 m/s) plus:
- quiet_stance (weight QUIET_WEIGHT): once a stop has settled, joint and root stillness (locomotion_quality.quiet_stance).
  heading_rew gives full credit for stepping in place at a stop; scratch stage-1 runs kept walking at a 0 m/s command.
- stop_probability 0.3 -> STOP_PROBABILITY: more stop goals.
"""
from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.rewards.locomotion_quality import quiet_stance

from examples.experiments.steering.mlp_human_model_v3_teacher_se_v1_s3 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_teacher_se_v1_s3 import env_config as _s3_env_config

QUIET_WEIGHT = 0.2
STOP_PROBABILITY = 0.4


def env_config(robot_cfg, args):
    cfg = _s3_env_config(robot_cfg, args)
    cfg.control_components["steering"].stop_probability = STOP_PROBABILITY
    cfg.reward_components["quiet_stance"] = MdpComponent(
        compute_func=quiet_stance,
        dynamic_vars={"dof_vel": EnvContext.current.dof_vel,
                      "root_pos": EnvContext.current.root_pos,
                      "prev_root_pos": EnvContext.steering.prev_root_pos,
                      "tar_speed": EnvContext.steering.tar_speed,
                      "speed_transition": EnvContext.steering.speed_transition,
                      "dt": EnvContext.dt},
        static_params={"dof_vel_scale": 0.5, "root_vel_err_scale": 20.0, "weight": QUIET_WEIGHT})
    return cfg
