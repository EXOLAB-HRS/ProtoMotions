"""vic4c_TC_scratch_ffdistill_r1 (strategy C + teacher torque coach, stage C1, candidate 2026-11-03, hyojun): from scratch,
the A1a recipe (vic4c_TC_scratch_r1: v4 law frozen at the human impedance centre, a = 0; narrow se_v1_s1 commands
0.8-1.2 m/s, no stops, no turns) plus the feed-forward distillation of vic4c_TC_s5_from-s5_ffdistill_r1: the frozen
fixed-PD teacher se-v1-s5 runs on the same observations and the 10 feed-forward outputs are pulled toward
(teacher PD torque - own PD torque), weight 0.5; swing knee out of r_imp. Reason: vic_TC_scratch_r17 (strategy C, v1 law,
no coach) walked without falls but lost the knee at speed (HMV-Walk knee r 0.98 / 0.78 / 0.59 at 0.8 / 1.0 / 1.4 m/s),
the same missing knee torque the coach supplied in the s5 path (0.8-1.0 m/s knee r 0.70-0.78 -> 0.95-0.97).
Teacher checkpoint: env HC_FF_DISTILL_TEACHER (se-v1-s5 last.ckpt, SHA b6075429...).
"""
import dataclasses
import os
from pathlib import Path

from examples.experiments.steering import mlp_human_model_v3_vic4c_tc_scratch_r1 as _a1a
from examples.experiments.steering.mlp_human_model_v3_vic4c_tc_scratch_r1 import *  # noqa: F401,F403

FFD_WEIGHT = 0.5
GAIT_R_TABLE = Path(__file__).resolve().parents[4] / "share/human_ref/fukuchi2018_knee_ankle_by_speed.pt"


def env_config(robot_cfg, args):
    cfg = _a1a.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(cfg.control_components["steering"], v4_imp_skip_swing_knee=True,
                                                             gait_r_table=str(GAIT_R_TABLE))
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    cfg = _a1a.agent_config(robot_cfg, env_cfg, args)
    teacher = os.environ.get("HC_FF_DISTILL_TEACHER", "")
    if not teacher or not os.path.isfile(teacher):
        raise FileNotFoundError(f"HC_FF_DISTILL_TEACHER must name the fixed-PD teacher checkpoint, got {teacher!r}")
    cfg.ff_distill.weight = FFD_WEIGHT
    cfg.ff_distill.teacher_checkpoint = teacher
    cfg.desync_initial_timeouts = True
    return cfg
