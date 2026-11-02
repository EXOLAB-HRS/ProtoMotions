"""vic4c_TC_scratch_from-ffdistillr1_r2 (strategy C + teacher torque coach, stage C2, candidate 2026-11-03, hyojun):
continues vic4c_TC_scratch_ffdistill_r1 with the full command set of A1b (vic4c_TC_scratch_from-vic4cr1_r2, s5 commands
with stops and turns); law, coach (weight 0.5, se-v1-s5) and swing-knee r_imp rule as in C1.
"""
import dataclasses
import os
from pathlib import Path

from examples.experiments.steering import mlp_human_model_v3_vic4c_tc_scratch_from_vic4cr1_r2 as _a1b
from examples.experiments.steering.mlp_human_model_v3_vic4c_tc_scratch_from_vic4cr1_r2 import *  # noqa: F401,F403

FFD_WEIGHT = 0.5
GAIT_R_TABLE = Path(__file__).resolve().parents[4] / "share/human_ref/fukuchi2018_knee_ankle_by_speed.pt"


def env_config(robot_cfg, args):
    cfg = _a1b.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(cfg.control_components["steering"], v4_imp_skip_swing_knee=True,
                                                             gait_r_table=str(GAIT_R_TABLE))
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    cfg = _a1b.agent_config(robot_cfg, env_cfg, args)
    teacher = os.environ.get("HC_FF_DISTILL_TEACHER", "")
    if not teacher or not os.path.isfile(teacher):
        raise FileNotFoundError(f"HC_FF_DISTILL_TEACHER must name the fixed-PD teacher checkpoint, got {teacher!r}")
    cfg.ff_distill.weight = FFD_WEIGHT
    cfg.ff_distill.teacher_checkpoint = teacher
    cfg.desync_initial_timeouts = True
    return cfg
