"""teacher_cadence_v1 revision 26: r25 with one change, ankle_pushoff_prior added (weight ANKLE_WEIGHT, sigma ANKLE_SIGMA_DEG).

r25 (r22 + heading_rew vel_err_scale 20) fixed overspeed (1.18 -> 1.11 m/s) but ankle push-off weakened
(peak plantarflexion -14.9 deg, ROM 27.8 deg vs Lencioni -22.4 / 30.8). The AMP pack d0b8c80d itself has a weak
push-off (peak -13.5 deg, ROM 23.7 deg) while r22 reached 31.1 deg on the same plant, so the plant can do it and
AMP pulls toward the pack. The ankle prior (r24 settings) counters that; r24 changed three terms at once, this one.
Starts from r25 last.ckpt.
"""
from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r25 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r25 import (
    env_config as _parent_env_config, agent_config)  # noqa: F401
from protomotions.envs.context_views import EnvContext
from protomotions.envs.control.cadence_gait_prior import GAIT_PCT, GAIT_PCT_VALID, ankle_pushoff_prior
from protomotions.envs.mdp_component import MdpComponent

ANKLE_WEIGHT = 0.30
ANKLE_SIGMA_DEG = 10.0


def env_config(robot_cfg, args):
    cfg = _parent_env_config(robot_cfg, args)
    dofs = list(robot_cfg.kinematic_info.dof_names)
    cfg.reward_components["ankle_pushoff_prior"] = MdpComponent(
        compute_func=ankle_pushoff_prior,
        dynamic_vars={"dof_pos": EnvContext.current.dof_pos, "tar_speed": EnvContext.steering.tar_speed,
                      "gait_pct": GAIT_PCT, "gait_pct_valid": GAIT_PCT_VALID},
        static_params={"table_file": str(LENCIONI_TABLE),
                       "ankle_dofs": [dofs.index("L_Ankle_y"), dofs.index("R_Ankle_y")],
                       "sign": -1.0, "sigma_deg": ANKLE_SIGMA_DEG, "weight": ANKLE_WEIGHT})
    return cfg
