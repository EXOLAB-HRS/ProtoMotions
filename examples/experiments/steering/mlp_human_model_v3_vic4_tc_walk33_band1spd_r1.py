# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 band1spd (server slots A/B, 2026-11-04 KST, user: "고정 pd 0.2 남기기 ... 극복해야 하는 문제", "고정 pd는 vic에서 빼도록"):
the cleaned law of vic4_TC_s5_from-s5_ffdistill_r1 (beta 0 on every joint: no fixed-PD torque in the executed law; stiffness
learned; feed-forward distillation from the frozen v3.3 walker is a training-only target) on plant v3.3, with
1. v4_band_margin 1.0 (was 2.0): stiffness may move only inside the human range. Night 11-03 runs at margin 2.0 put 43-68 %
   of the knee k / b outside the human band and the 1.4 m/s knee collapsed.
2. Command range 0.2-1.6 m/s with anchors (0.4, 1.0, 1.4, 1.6) (was 0.2-1.4, anchors 0.4/1.0/1.4): 1.4 m/s was the edge.
3. Speed reward without the soft underspeed tail (underspeed_tail_weight 0 in both speed terms): diagnosis 11-04 showed the
   policies walk at human cadence (121 steps/min) with 10-20 % short strides; the tail (scale 4) made going slow cheap.
   Overspeed keeps its stricter scale 40.
Diagnosis (logs/188_diag_speed14b.log): frozen stiffness (vic33c last) reaches only 1.22 / 1.33 m/s at commands 1.6 / 1.8,
so frozen human-centre stiffness is a capacity limit at 1.4; knee beta 0.2 reaches 1.44 / 1.61 (shortfall partly a reward
trade-off). Hence: stiffness open within the human band, plus the speed fixes.
"""
import dataclasses

from examples.experiments.steering import mlp_human_model_v3_vic4_tc_s5_from_s5_ffdistill_r1 as _base
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_s5_from_s5_ffdistill_r1 import *  # noqa: F401,F403


def env_config(robot_cfg, args):
    cfg = _base.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(
        cfg.control_components["steering"], v4_band_margin=1.0, beta_hold_joints=(), beta_hold=0.0,
        tar_speed_max=1.6, speed_anchor_targets=(0.4, 1.0, 1.4, 1.6))
    for k in ("speed_transition_tracking", "steady_speed_tracking"):
        if k in cfg.reward_components:
            cfg.reward_components[k].static_params["underspeed_tail_weight"] = 0.0
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _base.agent_config(robot_cfg, env_cfg, args)
