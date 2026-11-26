# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""vic4 v36 strat spd up abd + v4_metab_ce + symmetry + push-off / stiffness / metabolic / high-speed fixes (2026-11-15,
human-controller local v4.0 residual teacher run 5, user: "run5에 다 포함해서 하자"). The abd_sym_v40ce_r1 recipe plus:

- push-off: v4_strategy_band_k 1.0 -> HC_POF_BAND_K (0.5). Run 2 push-off 1.25 N m/kg sat just under the band edge
  (1.0 m/s mean - 1 SD = 1.33), so the one-sided ankle cost was ~0.07; 0.5 SD puts the edge at 1.41 and also narrows the
  two-sided hip band (late-stance hip -0.53 vs human -0.27).
- stiffness: v4_imp_see_series=True. The ankle k_eff of the impedance prior / W3 is the CE PD stiffness in series with
  the tendon's local slope, k_p k_t / (k_p + k_t) + k_pass, not k_p + k_pass (the plant's real joint stiffness on v4.0).
- metabolic proxy: vic_metab cost weight HC_METAB_WEIGHT (0.1, server 258 setting that passed W1/W2/W3 with the best
  W2 of its series). In run 2 v4_metab_ce was set but no reward read vic_metab; with it the ankle term is CE power, so
  Achilles recoil is free and CE work is charged.
- high-speed tracking: steady_speed_tracking underspeed_tail_weight 0 -> HC_UNDER_TAIL_W (0.3, tail scale 4). Run 2 last
  ran 1.24-1.26 m/s at a 1.4 command (W1 speed bias 0.136 > 0.05); exp(-200 e^2) is ~0.02 there with no gradient.
"""
import dataclasses
import os

from protomotions.envs.control.vic_steering import VIC_METAB, vic_cost
from protomotions.envs.mdp_component import MdpComponent

from examples.experiments.steering import mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_up_abd_sym_v40ce_r1 as _sym
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_walk33_v36_strat_spd_up_abd_sym_v40ce_r1 import *  # noqa: F401,F403

POF_BAND_K = float(os.environ.get("HC_POF_BAND_K", "0.5"))
METAB_WEIGHT = float(os.environ.get("HC_METAB_WEIGHT", "0.1"))
UNDER_TAIL_W = float(os.environ.get("HC_UNDER_TAIL_W", "0.3"))
# 2026-11-25 (server): joint-angle human prior weight; unset keeps the inherited weight (0.25). v4.0 students follow the
# teacher's W2 v5 (I e122 0.750/0.815 -> student 0.792/0.889 vs armsym1 0.833/0.852 -> 0.896/0.926), so the teacher's W2 is raised.
GAIT_PRIOR_W = os.environ.get("HC_GAIT_PRIOR_W")


def env_config(robot_cfg, args):
    cfg = _sym.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(
        cfg.control_components["steering"], v4_strategy_band_k=POF_BAND_K, v4_imp_see_series=True)
    cfg.reward_components["vic_metab"] = MdpComponent(
        compute_func=vic_cost, dynamic_vars={"value": VIC_METAB}, static_params={"weight": METAB_WEIGHT})
    cfg.reward_components["steady_speed_tracking"].static_params["underspeed_tail_weight"] = UNDER_TAIL_W
    if GAIT_PRIOR_W is not None:
        cfg.reward_components["human_gait_prior"].static_params["weight"] = float(GAIT_PRIOR_W)
    return cfg
