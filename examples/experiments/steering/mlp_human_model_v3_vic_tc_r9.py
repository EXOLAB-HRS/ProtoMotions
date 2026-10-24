"""vic_TC_r9 (implementation vic_tc_v1, candidate change, 2026-10-24): vic_TC_r6 plus a baseline-gait anchor.

r2-r8 (alpha = 1, beta = 0 fixed, from r1) all lost W1/W2 against r25 (falls 0.38-0.50/min, knee/ankle curves
collapsed) whatever the cost terms. r9 adds baseline_gait_prior: the human_gait_prior kernel against the fixed-PD
teacher's own gait (share/human_ref/baseline_gait_r25_teacher_cadence_v1.pt, r25 steady walking at 0.8 / 1.0 / 1.4 m/s,
Hip/Knee/Ankle mean and SD per gait percent), always on (no cadence-ratio gate), kernel 1 SD with a 2 deg floor.
It tests whether anchoring to the baseline gait keeps W1/W2 while VIC keeps W3/W4 (r6 passes both).
Starts from r1 last.ckpt like r2-r8. ANCHOR_WEIGHT is the only difference to r10.
"""
from pathlib import Path

from examples.experiments.steering.mlp_human_model_v3_vic_tc_r6 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_vic_tc_r6 import (
    env_config as _r6_env_config, agent_config)  # noqa: F401
from protomotions.envs.control.cadence_gait_prior import human_gait_prior
from protomotions.envs.mdp_component import MdpComponent

ANCHOR_WEIGHT = 0.25
ANCHOR_TABLE = Path(__file__).resolve().parents[4] / "share/human_ref/baseline_gait_r25_teacher_cadence_v1.pt"


def anchored_env_config(robot_cfg, args, weight):
    cfg = _r6_env_config(robot_cfg, args)
    prior = cfg.reward_components["human_gait_prior"]
    p = prior.static_params
    cfg.reward_components["baseline_gait_prior"] = MdpComponent(
        compute_func=human_gait_prior, dynamic_vars=dict(prior.dynamic_vars),
        static_params={"table_file": str(ANCHOR_TABLE), "left_dofs": list(p["left_dofs"]),
                       "right_dofs": list(p["right_dofs"]), "signs": list(p["signs"]), "sd_floor_deg": 2.0,
                       "kernel_scale": 1.0, "ratio_tol": 0.0, "terminal_knee_weight": 1.0,
                       "weight": weight})
    return cfg


def env_config(robot_cfg, args):
    return anchored_env_config(robot_cfg, args, ANCHOR_WEIGHT)
