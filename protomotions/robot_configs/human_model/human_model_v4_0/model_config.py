"""human_model_v4.0 (candidate, 2026-11-14; user: "joint level plant에 tendon넣은거를 v4.0이라고 하고 일단 모델은 만들어두자,
좀 문헌들 조사 잘 해서 사람데이터에 맞게 구현해 놔줘"): human_model_v3.6 + an ankle series elastic element (SEE).

The name human_model_v4.0 was used on 2026-11-11 for passive-moment candidates "B안/E안" (Silder knee-flexor /
gastrocnemius / end-range terms, jobs 7157/7178) that were trained and dropped; that code is not in this tree and is not
reused. From 2026-11-14 human_model_v4.0 means THIS tendon plant. Everything except the ankle actuator path is v3.6:
assets and segment inertia (v3.5 de Leva), ROM, passive exponentials (v3.4 set), strength ceilings (v3.6 young-adult MVC,
ankle plantarflexion 150 N m), PD defaults, collision. Model in common/series_elastic.py; runtime hook in
common/dynamics.py (implicit-PD path, HC_HUMAN_IMPLICIT_PD=1 required).

SEE parameters (63.3 kg model; tendon-level numbers -> joint level k = k_tendon r^2, D_toe = eps_toe L / r):
  Achilles (PF channel)
    k_tendon 180 N/mm   Lichtwark & Wilson 2005 J Exp Biol 208:4715 Table 1: 188 +- 43 N/mm (IQR 145-231), hopping, whole
                        tendon + aponeurosis; Peltonen 2013 SpringerPlus 2:212: 193-207 N/mm (10-80 % MVC, rate-independent);
                        Lichtwark & Wilson 2006/2007 model value ~180 N/mm. Sensitivity 145 / 231.
    r 0.05 m            Maganaris, Baltzopoulos & Sargeant 1998 J Physiol 510:977: rest 44 mm (-15 deg) to 55 mm (+30 deg),
                        +22-27 % at MVC; constant 50 mm (loaded walking range). Rugg 1990: +20 % toward plantarflexion (not modelled).
    -> k = 450 N m/rad (7.1 N m/rad/kg); whole-ankle walking quasi-stiffness 202-246 N m/rad (Shamaei 2013 PLoS One
       8:e59993) is lower, as expected for tendon in series with muscle.
    L_slack 0.24 m      whole tendon MTJ-calcaneus, from Lichtwark & Wilson 2005 (E A / k = 0.87 GPa x 53 mm^2 / 188 N/mm
                        = 245 mm) and 2006 (10.9 mm / 4.6 % = 237 mm). Used for the toe region and strain reporting only.
    eps_toe 0.02        toe-region strain ~2-3 % (in vitro; Thelen 2003 J Biomech Eng 125:70 eps_toe = 0.609 x 0.033 = 0.020).
                        -> D_toe = 0.096 rad, toe ends at 21.6 N m (432 N).
    tau_T 3 ms          tendon damping c_T = tau_T k = 1.35 N m s/rad: ~7 % of the stored energy lost per stance at 1.0 m/s
                        (validate_see.py), within in-vivo / in-vitro hysteresis 5-10 % (Peltonen 2013 5 %; Zelik & Franz 2017
                        PLoS One 12:e0179976 2-11 % at 1.25 m/s; Maganaris & Paul 2002 18 %, Lichtwark & Wilson 2005 26 % are
                        upper reports). Rate-dependent stand-in for a rate-independent loop.
  Tibialis anterior (DF channel; needed so the ankle has no PhysX drive at all, see dynamics.py)
    k_tendon 129 N/mm   Maganaris & Paul 1999 J Physiol 521:307: 530 N at 4.1 mm (secant; tangent 161 N/mm at max, still toe).
    r 0.04 m            Maganaris 2000 J Biomech 33:375: rest 4.5 (dorsiflexed) -> 2.9 cm (plantarflexed); +0.9-1.5 cm at MVC.
    -> k = 206 N m/rad, linear (toe 0). L_slack 0.164 m (4.1 mm / 2.5 %) for strain reporting only. tau_T 3 ms.
Knee / hip: no SEE. Series elasticity is minor in walking there (Sawicki, Lewis & Ferris 2009 Exerc Sport Sci Rev 37:130:
knee / hip apparent efficiency ~0.24 = muscle-like, ankle ~0.61 from Achilles recoil).
"""
from copy import deepcopy

MODEL = {
    "model_id": "human_model_v4.0", "resource_directory": "human_model_v4_0",
    "profile_id": "human_model_v3", "status": "candidate", "runnable": True,
    "assets": {"usd": "../human_model_v3_5/assets/human_model_v3_5.usda", "mjcf": "../human_model_v3_5/assets/human_model_v3_5.xml"},
    "profile": "../human_model_v3/profiles/healthy_adult_v3.json",
    "kinematic_model_id": "human_model_v2",
}

TAU_T_S = 0.003
ACHILLES = dict(k_tendon_n_per_m=180e3, moment_arm_m=0.05, slack_length_m=0.24, toe_strain=0.02)
TIBIALIS_ANTERIOR = dict(k_tendon_n_per_m=129e3, moment_arm_m=0.04, slack_length_m=0.164, toe_strain=0.0)


def series_elastic(achilles=None, tibialis=None, tau_t_s=TAU_T_S):
    """Serialized SEE config (stored in robot.human_model_parameters["series_elastic"], hence in checkpoints)."""
    from ..common.series_elastic import channel_from_tendon
    at, ta = dict(ACHILLES, **(achilles or {})), dict(TIBIALIS_ANTERIOR, **(tibialis or {}))
    pf = channel_from_tendon(+1, at["k_tendon_n_per_m"], at["moment_arm_m"], at["slack_length_m"], at["toe_strain"],
                             tau_t_s * at["k_tendon_n_per_m"] * at["moment_arm_m"] ** 2)
    df = channel_from_tendon(-1, ta["k_tendon_n_per_m"], ta["moment_arm_m"], ta["slack_length_m"], ta["toe_strain"],
                             tau_t_s * ta["k_tendon_n_per_m"] * ta["moment_arm_m"] ** 2)
    return {"model": MODEL["model_id"], "joints": ["L_Ankle_y", "R_Ankle_y"], "channels": {"PF": pf, "DF": df},
            "cap_at_ce_velocity": True}


def active_strength_scale():
    from ..human_model_v3_6.model_config import active_strength_scale as s36
    return s36()


def passive_exponentials():
    from ..human_model_v3_6.model_config import passive_exponentials as pe36
    return pe36()


def robot_config():
    from ..human_model_v3_6.model_config import robot_config as v36_config
    from ..common.integration import apply_metadata
    robot = v36_config()
    robot.human_model_parameters = dict(robot.human_model_parameters, series_elastic=deepcopy(series_elastic()))
    apply_metadata(robot)
    return robot
