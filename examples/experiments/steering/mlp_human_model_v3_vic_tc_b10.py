"""vic_TC_b10 (implementation vic_tc_v1 on vic_tc_v1_se, candidate, 2026-10-30): HMV_v1 path B diagnostic run. vic_TC_b8
last.ckpt (6393a916..., 60M) continued with only the quiet_stance reward of vic_TC_b9 (weight 0.2, dof_vel_scale 0.5 rad/s,
root_vel_err_scale 20; active once a 0 m/s command has settled). No sensory delay; gains, commands and other rewards as b8.

Why: b9 (b8 + quiet_stance + actor sensory delay 40-70 / 30-60 ms) collapsed by 10M (quiet standing 0-1 of 64, 33-45 of 64
steady envs fell) and the two changes cannot be told apart (Codex 2026-10-30). b10 tests the reward alone; the delay is
not dropped from path B, it is to be reintroduced after this and the paired b8 delay evaluation (0 vs 55/45 ms).
"""
from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.rewards.locomotion_quality import quiet_stance

from examples.experiments.steering import mlp_human_model_v3_vic_tc_b8 as _b8
from examples.experiments.steering.mlp_human_model_v3_vic_tc_b8 import *  # noqa: F401,F403

QUIET_WEIGHT = 0.2


def env_config(robot_cfg, args):
    cfg = _b8.env_config(robot_cfg, args)
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
