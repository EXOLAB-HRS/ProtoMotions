"""teacher_cadence_v1 revision 17: r16 (Lencioni pack, AMP weight 0.4) plus r14's knee_strike_prior
(no cadence gate, gait % > 85 or < 5, target 5 deg, weight 0.30). pelvis_height_prior is not added.
Starts from r15 last.ckpt, like r16, so r16 vs r17 isolates the knee term.
"""
from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r16 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r16 import (
    env_config as _r16_env_config, agent_config)  # noqa: F401
from protomotions.envs.context_views import EnvContext
from protomotions.envs.control.cadence_gait_prior import GAIT_PCT, GAIT_PCT_VALID, knee_strike_prior
from protomotions.envs.mdp_component import MdpComponent


def env_config(robot_cfg, args):
    cfg = _r16_env_config(robot_cfg, args)
    dofs = list(robot_cfg.kinematic_info.dof_names)
    cfg.reward_components["knee_strike_prior"] = MdpComponent(
        compute_func=knee_strike_prior,
        dynamic_vars={"dof_pos": EnvContext.current.dof_pos, "gait_pct": GAIT_PCT,
                      "gait_pct_valid": GAIT_PCT_VALID, "tar_cadence_ratio": EnvContext.steering.tar_cadence_ratio},
        static_params={"knee_dofs": [dofs.index("L_Knee_y"), dofs.index("R_Knee_y")], "ratio_tol": 0.0,
                       "late_pct": 0.85, "early_pct": 0.05, "target_deg": 5.0, "sigma_deg": 8.0, "weight": 0.30})
    return cfg
