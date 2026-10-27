"""vic2_TC_s5_r1 (implementation vic_tc_v2 on vic_tc_v1_se, candidate, 2026-10-28; v2 approved by the user).

Why v2: in vic_tc_v1 (vic_TC_s5_r15 / r17) the stiffness came from the state table alone. The policy's 10 per-joint
residuals stayed at delta ~ 0 (x1.0-1.15) in walking, standing and after pushes, k did not change with speed, and W3
"band" passed by construction. User (2026-10-28): stiffness should not jitter like a trajectory but separate coarsely
into swing / stance / both feet / standing / disturbance response, inside a physiologically reasonable range.

What changes against vic_TC_s5_r17 (r15 continued, actor lr 2e-5, human cadence law 0.558 v + 1.060):
1. Stiffness output (VicSteering coact_per_leg): one co-activation level u per leg, delta_j = h_j(state) * u,
   h = half log-width of the human range of the joint's state (delta_band_margin 1: never outside the range),
   activation-like filter 50 ms up / 150 ms down (coarse, no jitter). Cocontraction cost = relu(u) (weight 0.03).
2. A reason to stiffen: velocity pushes during training (PUSH_INTERVAL s apart, up to PUSH_LIN m/s horizontally),
   comparable to the evaluation battery (walk 0.2-0.5 m/s, stand 0.3-0.5 m/s).
3. Basics the user asked for: left-right gait symmetry reward (gait_symmetry.VicSymSteering, legs + arms at the same
   gait percent, SYM_WEIGHT) and the upper arms back in the AMP discriminator (s5amp "prox": + L/R Thorax and
   Shoulder; elbows / hands stay out because their passive joints wobble). Stability, speed / heading / cadence
   tracking, human joint-angle, knee-strike and pelvis priors and the VIC effort / band costs are r17's.
Starts from vic_TC_s5_r17 epoch 190 (best checkpoint: 4/16 W1-W4 rows fail, cadence / W3 / W4 pass); the launcher maps
the 79 v1 actor outputs to 59 + 2 + 10 (u at 0 = centre of the human range, feed-forward rows kept) and widens the
discriminator with zero-weight arm columns, so the run starts from the parent's behaviour.
"""
import dataclasses

from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.control.gait_symmetry import GAIT_SYMMETRY, VicSymSteeringConfig, gait_symmetry_reward
from protomotions.simulator.base_simulator.config import DomainRandomizationConfig, PushDomainRandomizationConfig

from examples.experiments.steering import mlp_human_model_v3_teacher_se_v1_s5amp as _s5amp
from examples.experiments.steering.mlp_human_model_v3_vic_tc_s5_r17 import *  # noqa: F401,F403
from examples.experiments.steering import mlp_human_model_v3_vic_tc_s5_r17 as _r17
from examples.experiments.steering.mlp_human_model_v3_vic_tc_s5_r17 import (
    env_config as _r17_env_config, agent_config as _r17_agent_config)

PUSH = True
PUSH_INTERVAL = (3.0, 6.0)
PUSH_LIN = (0.4, 0.4, 0.0)
ARMS_IN_DISC = True
SYM_WEIGHT = 0.2
COACT = dict(coact_per_leg=True, delta_band=True, delta_band_margin=1.0, delta_tau_up_s=0.05, delta_tau_down_s=0.15)


def configure_robot_and_simulator(robot_cfg, simulator_cfg, args):
    base = getattr(_r17, "configure_robot_and_simulator", None)      # stage_a's PD / plant setup
    if base is not None:
        base(robot_cfg, simulator_cfg, args)
    if PUSH:
        simulator_cfg.domain_randomization = DomainRandomizationConfig(push=PushDomainRandomizationConfig(
            push_interval_range=PUSH_INTERVAL, max_linear_velocity=PUSH_LIN, max_angular_velocity=(0.0, 0.0, 0.0)))


def env_config(robot_cfg, args):
    _s5amp.DISC_BODIES = "prox" if ARMS_IN_DISC else "legs"
    cfg = _r17_env_config(robot_cfg, args)
    old = cfg.control_components["steering"]
    kept = {f.name: getattr(old, f.name) for f in dataclasses.fields(old) if f.name != "_target_"}
    kept.update(COACT)
    cfg.control_components["steering"] = VicSymSteeringConfig(**kept)
    if SYM_WEIGHT > 0:
        cfg.reward_components["gait_symmetry"] = MdpComponent(
            compute_func=gait_symmetry_reward, dynamic_vars={"value": GAIT_SYMMETRY},
            static_params={"weight": SYM_WEIGHT})
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    _s5amp.DISC_BODIES = "prox" if ARMS_IN_DISC else "legs"
    cfg = _r17_agent_config(robot_cfg, env_cfg, args)
    n = robot_cfg.number_of_actions + 2 + len(VicSymSteeringConfig.vic_joints)
    cfg.model.actor.num_out = n
    cfg.model.actor.mu_model.num_out = n
    return cfg
