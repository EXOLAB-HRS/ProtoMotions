"""teacher_cadence_v1 revision 24: r22 (role split, AMP weight 0.4, knee_strike_prior weight 0.6) aimed at the three
remaining gaps against Lencioni et al. 2019:
- knee at contact (r21 18 deg vs human ~4 deg, same definition): knee_strike_prior sigma 8 -> KNEE_SIGMA_DEG;
- ankle push-off (Lencioni ankle r 0.71-0.84): ankle_pushoff_prior, weight ANKLE_WEIGHT, gait % 55-72,
  target = Lencioni table mean at the commanded speed and gait %, sigma ANKLE_SIGMA_DEG (6 deg gives 0.001 at a neutral ankle);
- step rate (r21 1.92 steps/s at 1 m/s vs f0 1.79): cadence_tracking relative scale 0.25 -> CADENCE_SCALE
  (an 8 % excess costs exp(-0.1) = 0.90 at 0.25, exp(-0.64) = 0.53 at 0.10). f0(v) is unchanged (Fukuchi fit, r9).
Starts from r22 last.ckpt.
"""
from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r22 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r22 import (
    env_config as _parent_env_config, agent_config)  # noqa: F401
from protomotions.envs.context_views import EnvContext
from protomotions.envs.control.cadence_gait_prior import GAIT_PCT, GAIT_PCT_VALID, ankle_pushoff_prior
from protomotions.envs.mdp_component import MdpComponent

KNEE_SIGMA_DEG = 5.0
ANKLE_WEIGHT = 0.30
CADENCE_SCALE = 0.10
ANKLE_SIGMA_DEG = 10.0


def env_config(robot_cfg, args):
    cfg = _parent_env_config(robot_cfg, args)
    dofs = list(robot_cfg.kinematic_info.dof_names)
    cfg.reward_components["knee_strike_prior"].static_params["sigma_deg"] = KNEE_SIGMA_DEG
    cfg.reward_components["cadence_tracking"].static_params["relative_scale"] = CADENCE_SCALE
    cfg.reward_components["ankle_pushoff_prior"] = MdpComponent(
        compute_func=ankle_pushoff_prior,
        dynamic_vars={"dof_pos": EnvContext.current.dof_pos, "tar_speed": EnvContext.steering.tar_speed,
                      "gait_pct": GAIT_PCT, "gait_pct_valid": GAIT_PCT_VALID},
        static_params={"table_file": str(LENCIONI_TABLE),
                       "ankle_dofs": [dofs.index("L_Ankle_y"), dofs.index("R_Ankle_y")],
                       "sign": -1.0, "sigma_deg": ANKLE_SIGMA_DEG, "weight": ANKLE_WEIGHT})
    return cfg
