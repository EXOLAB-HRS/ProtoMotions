"""vic_TC_r2 (implementation vic_tc_v1, candidate change, 2026-10-23): vic_TC_r1 continued with the fixes from the
r1 follow-up tests (human-controller output/261023/e01_vic_tc_r1_tests). Starts from r1 last.ckpt with alpha = 1,
beta = 0 held (no schedule).

- Start-up stiffness: contact on the first step after a reset comes from foot heights (fresh_stance), so the knee
  no longer starts at swing stiffness (r1: 0.37 vs r25 0.24 falls/min, fixed in evaluation alone).
- Foot impact: no cost. r1's 3-4 body-weight step peak was the substep (240 Hz) rigid-contact spike; at 30 Hz, close to
  force-plate filtering, r1 peaks at 1.29 and r25 at 1.18 BW (human 1.1-1.3). VicSteering keeps the impact input.
- Metabolic proxy of the sagittal leg joints (P+/0.25 + |P-|/1.20 in m*g*1 m/s). r1 dropped more human effort than
  a hip exo supplied (energy ratio 4.0-4.5, Collins limit about 4) with only a torque^2 cost.
- Speed tracking: r1 speed error 0.150 vs r25 0.089 m/s; heading_rew weight x HEADING_GAIN.
"""
import dataclasses

from examples.experiments.steering.mlp_human_model_v3_vic_tc_r1 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_vic_tc_r1 import (
    env_config as _r1_env_config, agent_config)  # noqa: F401
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.control.vic_steering import vic_cost, VIC_METAB

R2_WEIGHTS = {"vic_metab": (VIC_METAB, 0.03)}
HEADING_GAIN = 1.2


def env_config(robot_cfg, args):
    cfg = _r1_env_config(robot_cfg, args)
    old = cfg.control_components["steering"]
    cfg.control_components["steering"] = dataclasses.replace(
        old, alpha_init=1.0, beta_init=0.0, schedule=False, fresh_stance=True)
    for name, (path, weight) in R2_WEIGHTS.items():
        cfg.reward_components[name] = MdpComponent(
            compute_func=vic_cost, dynamic_vars={"value": path}, static_params={"weight": weight})
    cfg.reward_components["heading_rew"].static_params["weight"] *= HEADING_GAIN
    return cfg
