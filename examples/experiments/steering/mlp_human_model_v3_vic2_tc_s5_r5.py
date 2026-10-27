"""vic2_TC_s5_r5 (vic_tc_v2 candidate, 2026-10-28, local PC): vic2_TC_s5_r3 with stronger training pushes, up to 0.6 m/s
(r1 / r3: 0.4 m/s), same 3-6 s interval. Why: r3 (r1 recipe from vic_TC_s5_r15 last, 40M, 2048 env) left the co-activation
level flat after pushes (W3 v2 coact_push_rise 0.001, needs >= 0.2; training cocontraction ~0.034 throughout), so 0.4 m/s
pushes did not give the policy a reason to stiffen. 0.6 m/s covers the evaluation battery (walk 0.2-0.5, stand 0.3-0.5 m/s)
inside the training range. Everything else is r1's; start from vic_TC_s5_r15 last (as r3) so that the push strength is the
only difference to r3.
"""
from protomotions.simulator.base_simulator.config import DomainRandomizationConfig, PushDomainRandomizationConfig

from examples.experiments.steering import mlp_human_model_v3_vic2_tc_s5_r1 as _r1
from examples.experiments.steering.mlp_human_model_v3_vic2_tc_s5_r1 import *  # noqa: F401,F403

PUSH_LIN = (0.6, 0.6, 0.0)


def configure_robot_and_simulator(robot_cfg, simulator_cfg, args):
    _r1.configure_robot_and_simulator(robot_cfg, simulator_cfg, args)
    simulator_cfg.domain_randomization = DomainRandomizationConfig(push=PushDomainRandomizationConfig(
        push_interval_range=_r1.PUSH_INTERVAL, max_linear_velocity=PUSH_LIN, max_angular_velocity=(0.0, 0.0, 0.0)))
