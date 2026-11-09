"""Walk teacher on human_model_v3.6: vic4_tc_walk33_v36_strat_spd_up_r1 (the 344 recipe) with the joint-angle reward on WBDS train5.

User 2026-11-09: rank by W2 (joint angles) and W2-2k (joint torque) before W1, follow the push-off lineage, 100M runs. The 344
recipe gives the best kinetics so far (344 last push-off 1.61 / human 1.43, late-stance hip -0.35 / -0.26, W2 v5 all seeds).
This variant only swaps the joint-angle reward table to the WBDS train5 profile, the same dataset (other subjects) as the W2
WBDS rows and the W2-2k moments, so angles and torques are trained against one experiment:
  human_gait_prior table_file: lencioni2019_hip_knee_ankle_by_speed_holdout5.pt -> wbds2018_hip_knee_ankle_by_speed_stand_train5.pt.
"""
from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_up_r1 as _base
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_up_r1 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_v35_strat_spd_wang_r1 import ANGLE_TABLE


def env_config(robot_cfg, args):
    cfg = _base.env_config(robot_cfg, args)
    cfg.reward_components["human_gait_prior"].static_params["table_file"] = str(ANGLE_TABLE)
    return cfg
