"""vic2_TC_s5_r9 (vic_tc_v2 candidate, 2026-10-28, local PC): vic2_TC_s5_r6 (u log std -1.2) with r5's stronger training
pushes, up to 0.6 m/s (r6: 0.4 m/s), same 3-6 s interval. Why: r6 explored u (sigma 0.30) but still kept it flat after
pushes (coact_push_rise 0.0006); at 0.4 m/s the training fall rate stayed ~0.0003 per step, so stiffening never paid. r5's
0.6 m/s pushes did raise falls, but there u was barely explored (sigma 0.055). r9 puts both together; start from
vic_TC_s5_r15 last as r3 / r5 / r6.
"""
from protomotions.simulator.base_simulator.config import DomainRandomizationConfig, PushDomainRandomizationConfig

from examples.experiments.steering.mlp_human_model_v3_vic2_tc_s5_r6 import *  # noqa: F401,F403

from examples.experiments.steering import mlp_human_model_v3_vic2_tc_s5_r1 as _r1

PUSH_LIN = (0.6, 0.6, 0.0)


def configure_robot_and_simulator(robot_cfg, simulator_cfg, args):
    _r1.configure_robot_and_simulator(robot_cfg, simulator_cfg, args)
    simulator_cfg.domain_randomization = DomainRandomizationConfig(push=PushDomainRandomizationConfig(
        push_interval_range=_r1.PUSH_INTERVAL, max_linear_velocity=PUSH_LIN, max_angular_velocity=(0.0, 0.0, 0.0)))
