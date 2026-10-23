"""vic_TC_r4 (implementation vic_tc_v1, candidate change, 2026-10-24, local gunhee2): vic_TC_r2 with stronger speed
tracking, run locally in parallel with the server's r2 and r3. Starts from r1 last.ckpt like r2.

Speed is the r1 failure that r2 and r3 address least: r1 speed error 0.150 vs r25 0.089 m/s (5 min endurance,
human-controller output/261023/e01_vic_tc_r1_tests). The local damping-only control (r25 fine-tuned with leg Kd x0.25,
human-controller output/261023/e02_teacher_r25_physio_kd025_local) shows the same pattern on a softer plant: v/v* 1.19-1.42
at 0.8-1.0 m/s (r25 1.09-1.12) and 0.6 m/s unstable (0.66-1.38). Two changes against r2:
- heading_rew weight x HEADING_GAIN over r25 (r2: x1.2). Steady-state speed is scored only by heading_rew (r25 note).
- speed anchors SPEED_ANCHORS: 0.6 and 0.8 m/s added to r25's (0.4, 1.0, 1.4), anchor probability unchanged (0.5),
  so the slow band where both soft plants drift gets explicit samples (each anchor 10 % of non-stop resamples).
"""
import dataclasses

from examples.experiments.steering.mlp_human_model_v3_vic_tc_r2 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_vic_tc_r2 import (
    env_config as _r2_env_config, agent_config, HEADING_GAIN as _R2_HEADING_GAIN)  # noqa: F401

HEADING_GAIN = 1.5
SPEED_ANCHORS = (0.4, 0.6, 0.8, 1.0, 1.4)


def env_config(robot_cfg, args):
    cfg = _r2_env_config(robot_cfg, args)
    cfg.reward_components["heading_rew"].static_params["weight"] *= HEADING_GAIN / _R2_HEADING_GAIN
    old = cfg.control_components["steering"]
    cfg.control_components["steering"] = dataclasses.replace(old, speed_anchor_targets=SPEED_ANCHORS)
    return cfg
