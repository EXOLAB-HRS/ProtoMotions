"""teacher_cadence_v1 revision 13: r12 plus a heel-strike knee term and a pelvis-height term.

r12 against Lencioni et al. 2019: hip/ankle/step rate in the human range, knee not (r 0.872,
landing at ~22 deg flexion vs human ~4 deg, still extending at contact; pelvis 0.942 m walking vs
0.975 m standing). Both terms only at natural cadence (|rho* - 1| < 0.05); existing weights unchanged.
- knee_strike_prior, weight KNEE_STRIKE_WEIGHT: knee near TARGET_DEG in gait % > 90 or < 3;
- pelvis_height_prior, weight PELVIS_WEIGHT: pelvis height near PELVIS_TARGET m.
PELVIS_TARGET is standing height minus 1 cm (assumed, not measured from human data).
Starts from r12 last.ckpt.
"""
from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r12 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r12 import (
    env_config as _r12_env_config, agent_config)  # noqa: F401
from protomotions.envs.context_views import EnvContext
from protomotions.envs.control.cadence_gait_prior import (
    GAIT_PCT, GAIT_PCT_VALID, knee_strike_prior, pelvis_height_prior)
from protomotions.envs.mdp_component import MdpComponent

KNEE_STRIKE_WEIGHT = 0.15
TARGET_DEG = 5.0
PELVIS_WEIGHT = 0.05
PELVIS_TARGET = 0.965


def env_config(robot_cfg, args):
    cfg = _r12_env_config(robot_cfg, args)
    dofs = list(robot_cfg.kinematic_info.dof_names)
    ratio = EnvContext.steering.tar_cadence_ratio
    cfg.reward_components["knee_strike_prior"] = MdpComponent(
        compute_func=knee_strike_prior,
        dynamic_vars={"dof_pos": EnvContext.current.dof_pos, "gait_pct": GAIT_PCT,
                      "gait_pct_valid": GAIT_PCT_VALID, "tar_cadence_ratio": ratio},
        static_params={"knee_dofs": [dofs.index("L_Knee_y"), dofs.index("R_Knee_y")],
                       "target_deg": TARGET_DEG, "weight": KNEE_STRIKE_WEIGHT})
    cfg.reward_components["pelvis_height_prior"] = MdpComponent(
        compute_func=pelvis_height_prior,
        dynamic_vars={"root_height": EnvContext.current.root_height,
                      "tar_speed": EnvContext.steering.tar_speed, "tar_cadence_ratio": ratio},
        static_params={"target": PELVIS_TARGET, "weight": PELVIS_WEIGHT})
    return cfg
