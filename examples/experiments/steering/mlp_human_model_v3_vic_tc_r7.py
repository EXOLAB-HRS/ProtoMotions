"""vic_TC_r7 (implementation vic_tc_v1, candidate change, 2026-10-24): vic_TC_r4 with the human gait prior active at
every commanded cadence and weighted x PRIOR_GAIN. Starts from r1 last.ckpt like r2-r6.

Joint curves at 1 m/s (right leg, every 10 % of the stride, vs Fukuchi 2018; scratch diagnosis of the W2 base runs):
the r1/r4 knee stays straight through stance (about 0 deg at 10-50 %, human 14 deg at 10-20 %, r25 9-15 deg) and the
r4 ankle plantarflexes in mid stance (-22 deg at 20 %, human +6) and in swing (-33 deg at 80 %, human +4). The prior
(Lencioni curves) already penalises this but only when |rho* - 1| < ratio_tol = 0.05 (r12, table measured at natural
cadence), i.e. in ~30 % of the episodes (cadence_natural_fraction 0.3). r7 sets ratio_tol to RATIO_TOL so the prior
covers the whole commanded range 0.8-1.25 (curves change little with cadence at a given speed) and raises its weight.
"""
from examples.experiments.steering.mlp_human_model_v3_vic_tc_r4 import *  # noqa: F401,F403
from examples.experiments.steering.mlp_human_model_v3_vic_tc_r4 import (
    env_config as _r4_env_config, agent_config)  # noqa: F401

PRIOR_GAIN = 1.6
RATIO_TOL = 0.30


def env_config(robot_cfg, args):
    cfg = _r4_env_config(robot_cfg, args)
    prior = cfg.reward_components["human_gait_prior"].static_params
    prior["weight"] *= PRIOR_GAIN
    prior["ratio_tol"] = RATIO_TOL
    return cfg
