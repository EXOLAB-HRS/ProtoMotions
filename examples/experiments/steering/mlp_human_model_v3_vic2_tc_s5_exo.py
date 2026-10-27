"""vic2_TC_s5 exo candidate (implementation vic_tc_v2 on vic_tc_v1_se, candidate change, 2026-10-28; no run number yet).

Why: in vic_TC_s5_r15 / r17 / r18 the W4 rows (hip-exo assist / resist response) got worse with longer training, and
the only training disturbance in vic2_TC_s5_r1 is a body push. The user asked to implement exoskeleton assist / resist
randomisation during training now and decide the run later (2026-10-28).

What changes against vic2_TC_s5_r1 (everything else, including the pushes, is r1's):
- VicSteering exo_rand: per env, every 4-8 s a new segment is off (30 %), resists (30 %) or assists (40 %) the hip with
  the W4 evaluation profile (extension bump on 0-30 % and flexion bump on 45-75 % of each leg's gait cycle on Hip_y),
  amplitudes 5-20 N m drawn per segment (evaluation uses 10 / 20 assist and 10 resist). Torque is added outside the
  muscle caps and is not observed by the policy.
The run name (vic2_TC_<origin>_r<n>) and the launcher entry are added when the run is assigned in the vic_TC_server table.
"""
from examples.experiments.steering.mlp_human_model_v3_vic2_tc_s5_r1 import *  # noqa: F401,F403
from examples.experiments.steering import mlp_human_model_v3_vic2_tc_s5_r1 as _r1

EXO = dict(exo_rand=True, exo_amp_range=(5.0, 20.0), exo_off_prob=0.3, exo_resist_prob=0.3,
           exo_interval_range=(4.0, 8.0), exo_min_speed=0.3)


def env_config(robot_cfg, args):
    cfg = _r1.env_config(robot_cfg, args)
    steering = cfg.control_components["steering"]
    for k, v in EXO.items():
        setattr(steering, k, v)
    return cfg
