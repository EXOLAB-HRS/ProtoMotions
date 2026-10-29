"""vic2_TC_s5_r10 (vic_tc_v2 candidate, 2026-10-28, local PC): vic2_TC_s5_r9 (u log std -1.2, training pushes up to
0.6 m/s) with the cocontraction cost weight 0.03 -> 0 (the term stays logged). Why: r3 / r5 / r6 / r9 all kept u flat after
pushes (coact_push_rise 0.001 / -0.002 / 0.0006 / 0.0001) and mean walking u went slightly negative (r6 -0.10, r9 -0.06),
so the only reward that sees u may be the cost on relu(u). With it removed, u rising after pushes would mean the cost was
holding it down; u still flat would mean nothing in the reward pays for stiffening during a push. Start from
vic_TC_s5_r15 last as r3 / r5 / r6 / r9.
"""
from examples.experiments.steering.mlp_human_model_v3_vic2_tc_s5_r9 import *  # noqa: F401,F403

from examples.experiments.steering import mlp_human_model_v3_vic2_tc_s5_r9 as _r9

COCONTRACTION_WEIGHT = 0.0


def env_config(robot_cfg, args):
    cfg = _r9.env_config(robot_cfg, args)
    cfg.reward_components["vic_cocontraction"].static_params["weight"] = COCONTRACTION_WEIGHT
    return cfg
