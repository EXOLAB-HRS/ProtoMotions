"""a7-gp20: a7 warm start with the discriminator gradient penalty raised 5 -> 20.

Single-variable screening for the AMP saturation that every a7-lineage run shows.
a5, a5_disclr2e5, a9, a10 and b1-a7 all end with discriminator accuracy near
1.000 on reference and 0.981 on agent samples, so the style reward sits at the
floor (0.09-0.11 against a 0.39 maximum) and the policy is effectively trained by
task reward alone.  Measured consequence: within-stride forward-speed oscillation
of 0.039-0.050 m/s against a reference floor of 0.0858.

Lowering the discriminator learning rate was already screened and did not move the
accuracies (a5_disclr2e5: 1.000/0.981 against a5's 1.00/0.982).  The gradient
penalty limits how sharp the discriminator's decision boundary may be, which is
the term that bounds separating power rather than the speed of acquiring it.

Everything else is a7: plant, motion pack, Stage A commands, reward components and
weights, task/AMP 0.6/0.4, velocity error scale 2.0, learning rates, network, and
the PPO settings.  Runtime env/batch, seed and budget are launcher inputs and must
match the run this is compared against.

The screening decides on two measurements, not on the accuracies themselves:
  - Stage A must not regress (forward MAE <= 0.15 at 0.8/1.0/1.2; a7 is
    0.0434/0.0668/0.0472).
  - Oscillation must move from 0.039-0.045 toward the reference band 0.086-0.162.

If the style reward recovers but the oscillation does not, the over-smooth gait is
a plant or actuation limit rather than a reward-shaping one, which is the more
useful outcome of the two.
"""
from examples.experiments.steering.mlp_human_model_v3_a7_followups import *
from examples.experiments.steering.mlp_human_model_v3_a7_followups import (
    agent_config as _a7_agent_config,
)

DISCRIMINATOR_GRAD_PENALTY = 20.


def env_config(robot_cfg, args):
    return make_env_config(robot_cfg, args, "a7")


def agent_config(robot_cfg, env_cfg, args):
    cfg = _a7_agent_config(robot_cfg, env_cfg, args)
    cfg.amp_parameters.discriminator_grad_penalty = DISCRIMINATOR_GRAD_PENALTY
    return cfg
