"""vic4_TC_s5_from-ramp2x_r2 (local, 2026-11-01; user "탐색보다 빨리 좋은 결과"): continuation of vic4_TC_s5_from-s5_noeff-ramp2x_r1
(seed 2944) from its last checkpoint after the user stop rule (e1281 knee r 0.918 / 0.720 / 0.666, recovering from e1098
0.670 / 0.538 / 0.447), at the final law with no ramp (alpha 1, beta 0, schedule off) so the reloaded ramp does not
restart. Same rewards and settings otherwise; only runs if the substep-prior run fails its post-ramp check.
"""
import dataclasses

from examples.experiments.steering import mlp_human_model_v3_vic4_tc_s5_from_s5_noeff_ramp2x_r1 as _r2x
from examples.experiments.steering.mlp_human_model_v3_vic4_tc_s5_from_s5_noeff_ramp2x_r1 import *  # noqa: F401,F403


def env_config(robot_cfg, args):
    cfg = _r2x.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(cfg.control_components["steering"],
                                                             alpha_init=1.0, beta_init=0.0, schedule=False)
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _r2x.agent_config(robot_cfg, env_cfg, args)
