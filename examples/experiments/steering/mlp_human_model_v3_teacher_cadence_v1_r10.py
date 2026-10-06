"""teacher_cadence_v1 revision 10: r9 plus a human knee/ankle gait-angle prior.

r8 compared with Fukuchi et al. 2018 treadmill walking (scripts/human_gait_ref_compare.py,
2026-10-06): hip r 0.96, knee 0.88, ankle 0.88, but no early-stance knee flexion and no
late-swing ankle dorsiflexion. The forward25 motion pack already lacks both, and the source
mocap's own correlation with the human data is below r8, so the AMP reference cannot add them.
- steering: CadenceGaitPercentSteering (adds per-leg gait percent from stance onsets; command,
  clock and other context unchanged);
- reward human_gait_prior, weight PRIOR_WEIGHT (added; existing weights unchanged): knee flexion
  and ankle dorsiflexion against the Fukuchi table by commanded speed and gait percent
  (share/human_ref/fukuchi2018_knee_ankle_by_speed.pt).
Trained against Fukuchi, so the human-likeness check of this run must use another dataset.
"""
import dataclasses
from pathlib import Path

from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r9 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r9 import (
    env_config as _r9_env_config, agent_config)  # noqa: F401
from protomotions.envs.context_views import EnvContext
from protomotions.envs.control.cadence_gait_prior import (
    CadenceGaitPercentSteeringConfig, GAIT_PCT, GAIT_PCT_VALID, human_gait_prior)
from protomotions.envs.mdp_component import MdpComponent

PRIOR_WEIGHT = 0.10
PRIOR_TABLE = Path(__file__).resolve().parents[4] / "share/human_ref/fukuchi2018_knee_ankle_by_speed.pt"


def env_config(robot_cfg, args):
    cfg = _r9_env_config(robot_cfg, args)
    base = cfg.control_components["steering"]
    cfg.control_components["steering"] = CadenceGaitPercentSteeringConfig(
        **{f.name: getattr(base, f.name) for f in dataclasses.fields(base) if f.name != "_target_"})
    dofs = list(robot_cfg.kinematic_info.dof_names)
    cfg.reward_components["human_gait_prior"] = MdpComponent(
        compute_func=human_gait_prior,
        dynamic_vars={"dof_pos": EnvContext.current.dof_pos, "tar_speed": EnvContext.steering.tar_speed,
                      "gait_pct": GAIT_PCT, "gait_pct_valid": GAIT_PCT_VALID},
        static_params={"table_file": str(PRIOR_TABLE),
                       "left_dofs": [dofs.index("L_Knee_y"), dofs.index("L_Ankle_y")],
                       "right_dofs": [dofs.index("R_Knee_y"), dofs.index("R_Ankle_y")],
                       "signs": [1.0, -1.0], "sd_floor_deg": 3.0, "weight": PRIOR_WEIGHT})
    return cfg
