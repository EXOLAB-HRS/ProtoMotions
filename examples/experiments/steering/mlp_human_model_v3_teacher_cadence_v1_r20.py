"""teacher_cadence_v1 revision 20: role split. AMP learns whole-body motion from natural AMASS clips;
the leg gait cycle is shaped toward human lab data by a reward.

r15-r19 used the Lencioni pack as the AMP reference: one stride repeated, straight walking only, mean
upper-body cycle. That is not natural motion for a human avatar (no turns, starts, stops, stride-to-stride
variation). r20:
- --motion = AMASS v4 train pack (100 clips: straight, turns, circles, stand-to-walk; sha d0b8c80d), with
  cadence labels built for it;
- human_gait_prior table = Lencioni 2019 adults, Hip/Knee/Ankle sagittal, standing-relative, by speed and
  gait percent; subjects with id % 5 == 0 held out (scripts/build_human_gait_table.py --ref lencioni);
  weight, kernel, terminal knee weight and cadence gate as r12;
- knee_strike_prior as r17 (no cadence gate, gait % > 85 or < 5, target 5 deg, weight 0.30);
- AMP reward weight AMP_REWARD_W (r12 value 0.1).
Starts from r17 last.ckpt (best leg kinematics so far).
"""
from pathlib import Path

from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r12 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r12 import (
    env_config as _r12_env_config, agent_config as _r12_agent_config)
import examples.experiments.steering.mlp_human_model_v3_c_a7_cadence as _c_a7_cadence
import examples.experiments.steering.mlp_human_model_v3_teacher_cadence_v1_r3 as _r3
from protomotions.envs.context_views import EnvContext
from protomotions.envs.control.cadence_gait_prior import GAIT_PCT, GAIT_PCT_VALID, knee_strike_prior
from protomotions.envs.mdp_component import MdpComponent

_SHARE = Path(__file__).resolve().parents[4] / "share"
# The cadence modules read their label files from module globals at call time.
_c_a7_cadence.LABEL_FILE = _SHARE / "cadence/amasstrain100_d0b8c80d_cadence_labels.pt"
_r3.R3_LABEL_FILE = _SHARE / "cadence/amasstrain100_d0b8c80d_cadence_labels_r080_125.pt"
LENCIONI_TABLE = _SHARE / "human_ref/lencioni2019_hip_knee_ankle_by_speed_holdout5.pt"
AMP_REWARD_W = 0.1


def env_config(robot_cfg, args):
    cfg = _r12_env_config(robot_cfg, args)
    dofs = list(robot_cfg.kinematic_info.dof_names)
    prior = cfg.reward_components["human_gait_prior"].static_params
    prior.update(table_file=str(LENCIONI_TABLE),
                 left_dofs=[dofs.index(d) for d in ("L_Hip_y", "L_Knee_y", "L_Ankle_y")],
                 right_dofs=[dofs.index(d) for d in ("R_Hip_y", "R_Knee_y", "R_Ankle_y")],
                 signs=[-1.0, 1.0, -1.0])
    cfg.reward_components["knee_strike_prior"] = MdpComponent(
        compute_func=knee_strike_prior,
        dynamic_vars={"dof_pos": EnvContext.current.dof_pos, "gait_pct": GAIT_PCT,
                      "gait_pct_valid": GAIT_PCT_VALID, "tar_cadence_ratio": EnvContext.steering.tar_cadence_ratio},
        static_params={"knee_dofs": [dofs.index("L_Knee_y"), dofs.index("R_Knee_y")], "ratio_tol": 0.0,
                       "late_pct": 0.85, "early_pct": 0.05, "target_deg": 5.0, "sigma_deg": 8.0, "weight": 0.30})
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    cfg = _r12_agent_config(robot_cfg, env_cfg, args)
    cfg.amp_parameters.discriminator_reward_w = AMP_REWARD_W
    return cfg
