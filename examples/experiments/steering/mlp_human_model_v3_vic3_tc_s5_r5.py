"""vic3_TC_s5_r5 (vic_tc_v3 candidate, 2026-10-29; user: run the tracking fix tonight) = vic3_TC_s5_r4 + efference copy:
the actor also sees its own last 3 actions (79 each: 59 PD targets, 10 u, 10 tau_ff; raw, undelayed; 237 inputs).
Why: since the afferent delay (r13 on) every run tracks speed / lateral at 0.13-0.17 / 0.11-0.15 m/s (r11 without delay
0.082 / 0.066); 20M more training (r13b) did not help, and the command preview (1.0 s, 0.5 s acceleration) was already
on. With delayed state alone the policy cannot tell what its last commands are doing yet. A person predicts the current
state from the delayed afferents plus the efference copy of the motor command (forward model; Wolpert & Miall 1996,
the CB role in this project). 3 steps = 100 ms >= the longest proprioceptive delay (70 ms) + one control step.
The launcher appends zero-weight actor input columns (append_efference_inputs), so the first actions equal r3's.
Critic and discriminators unchanged. Compare with vic3_TC_s5_r4 (same start, same rest).
"""
from protomotions.envs.component_factories import previous_actions_factory

from examples.experiments.steering import mlp_human_model_v3_vic3_tc_s5_r4 as _v3r4
from examples.experiments.steering.mlp_human_model_v3_vic3_tc_s5_r4 import *  # noqa: F401,F403

EFFERENCE_STEPS = 3
EFFERENCE_KEY = "efference_copy"


def env_config(robot_cfg, args):
    cfg = _v3r4.env_config(robot_cfg, args)
    cfg.observation_components[EFFERENCE_KEY] = previous_actions_factory(history_steps=EFFERENCE_STEPS)
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    cfg = _v3r4.agent_config(robot_cfg, env_cfg, args)
    m = cfg.model
    for module in (m.actor, m.actor.mu_model):
        module.in_keys = list(module.in_keys) + [EFFERENCE_KEY]
    m.in_keys = list(m.in_keys) + [EFFERENCE_KEY]
    return cfg
