"""vic_TC_b9 (implementation vic_tc_v1 on vic_tc_v1_se, candidate, 2026-10-30): HMV_v1 path B. vic_TC_b8 last.ckpt
(6393a916..., 60M) continued with two changes, everything else (gains incl. the b8 knee stance range, commands,
stop_probability 0.3, other rewards) as b8:
1. quiet_stance reward (teacher_se_v1_s3stop setting: weight 0.2, dof_vel_scale 0.5 rad/s, root_vel_err_scale 20),
   active only when a 0 m/s command has settled.
2. actor sensory delay from the VIC common update guide (stage_vic_common_update_261028.md item 1, B stage 1):
   proprio 40-70 ms, vestibular 30-60 ms, critic and discriminator unchanged (true values).

Why: b8 kept the best path-B gait (W2 v4 fails 5|6|6, hip offset and cadence pass at all speeds) but quiet standing
stayed 26-28 of 64. In the 2-3 s standing window the binding condition is leg joint speed > 0.5 rad/s, mostly the knee;
the knee target q_exec swings ~10-15 deg (SD) against ~2-5 deg of actual motion while vic_ff varies 1-3 N m, so the
standing motion comes from the policy's targets, not the residual torque (Claude and Codex, 2026-10-30). b4 had no
quiet_stance term. The two changes are not separable in this run (the guide asks for the delay in new runs).
"""
from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.obs.sensory_delay import SensoryDelayConfig, delay_actor_inputs
from protomotions.envs.rewards.locomotion_quality import quiet_stance

from examples.experiments.steering import mlp_human_model_v3_vic_tc_b8 as _b8
from examples.experiments.steering.mlp_human_model_v3_vic_tc_b8 import *  # noqa: F401,F403

QUIET_WEIGHT = 0.2
SENSORY_DELAY = dict(enabled=True, proprio_ms=(40.0, 70.0), vestib_ms=(30.0, 60.0))


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
    cfg.sensory_delay = SensoryDelayConfig(**SENSORY_DELAY)
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    cfg = _b8.agent_config(robot_cfg, env_cfg, args)
    return delay_actor_inputs(cfg, env_cfg.sensory_delay)
