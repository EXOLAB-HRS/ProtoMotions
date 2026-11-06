"""human_model_v3.6 (2026-11-07, user: "상한은 사람답지않개 막으면 안되지 플랜트 상한은 실제 사람이 낼수 있는 최대로 반영하는게",
"모든관절에대해 검토좀 다시 해줘"): human_model_v3.5 with joint torque ceilings raised to young-adult maximal voluntary
strength where the healthy_adult_v3 value (Danneskiold-Samsoe 2009 pooled 20-80 y and proxies) is below it.
Rule: a ceiling is the most a real person can produce; walking torque is shaped by rewards, not by the ceiling. Values are
raised only where two consistent sources agree (or a measured young-adult table); nothing is lowered here (values above
the evidence are listed as review items in STRENGTH_REVIEW). Mass properties (v3.5), geometry, ROM, passive terms and PD
gains are unchanged. Applied with human_model_parameters["active_strength_scale"] (common/dynamics.py), so control_info
effort limits follow; recipes pin the VIC torque normalisation to the v3.5 limits with VicSteering.v4_tau_norm.

Young adults (20-40 y), sex mean, isometric unless noted, N m (current -> v3.6):
  Knee extension   149  -> 200  Harbo 2012 regression 211 (30 y, 63 kg, 1.70 m), measured <30 y 225; Anderson 2007 170
  Ankle dorsiflex  23.7 -> 34   Harbo 2012 33 (regression), 36 (<30 y); Anderson 2007 32
  Ankle plantarflex 118.3 -> 150 straight-knee isometric MVC of young adults (~150-200 men, ~100-130 women; Harbo 103 is
                                isokinetic 60 deg/s, Anderson 105 is with the knee 50 deg flexed, both under the
                                straight-knee push-off condition). Least certain row.
  Subtalar inv/ev  18.2 -> 30   Biodex normative 2025 (World J Orthop 16:108858) inversion 33 / eversion 28; Ottaviani
                                (young women, weight-bearing) eversion 24. Same value both directions (upper bound).
  Toe (MTP) flexion 10 -> 15    Endo 2014 J Foot Ankle Res 7:26, young adults 19 +- 7 N m (first MTP, standing).
  Wrist flexion    9.26 -> 20   Harbo 2012 measured <30 y 25 / 15 (M / F); extension 6.70 -> 9 (Harbo 11 / 7).
"""
MODEL = {
    "model_id": "human_model_v3.6", "resource_directory": "human_model_v3_6",
    "profile_id": "human_model_v3", "status": "candidate", "runnable": True,
    "assets": {"usd": "../human_model_v3_5/assets/human_model_v3_5.usda", "mjcf": "../human_model_v3_5/assets/human_model_v3_5.xml"},
    "profile": "../human_model_v3/profiles/healthy_adult_v3.json",
    "kinematic_model_id": "human_model_v2",
}

# joint suffix -> {healthy_adult_v3 value: v3.6 value}; matched by value so the L / R sign mirroring needs no table
RAISE = {
    "Knee_y": {149.0: 200.0},
    "Ankle_y": {23.7: 34.0, 118.3: 150.0},
    "Subtalar_x": {18.1641504178273: 30.0},
    "Toe_y": {10.0: 15.0},
    "Wrist_x": {9.258426183844012: 20.0, 6.701337047353761: 9.0},
}
# above the young-adult evidence (kept as the ceiling; candidates if a later version tightens): hip extension 219.5
# (Anderson 180), hip adduction 170.5 (~95-140, recall), elbow flexion 46.9 (Harbo 36-40). Unverified (recall only):
# trunk lateral bend / rotation, hip and shoulder rotation, forearm, wrist deviation, girdle proxies.
STRENGTH_REVIEW = "output/261107 review in share/training_configs/vic_TC_b_local/log.md (2026-11-07)"


def active_strength_scale():
    import json
    from pathlib import Path
    p = json.loads((Path(__file__).resolve().parent / MODEL["profile"]).read_text())
    out = {}
    for name, row in p["joints"].items():
        table = next((v for k, v in RAISE.items() if name.endswith(k)), None)
        if table is None:
            continue
        f = [next((new / cur for old, new in table.items() if abs(cur - old) < 1e-6), 1.0) for cur in row["active_nm"]]
        assert max(f) > 1.0, (name, row["active_nm"])
        out[name] = f
    return out


def passive_exponentials():
    from ..human_model_v3_5.model_config import passive_exponentials as pe35
    return pe35()


def robot_config():
    from ..human_model_v3_5.model_config import robot_config as v35_config
    from ..common.integration import apply_metadata
    robot = v35_config()
    robot.human_model_parameters = dict(robot.human_model_parameters, active_strength_scale=active_strength_scale())
    apply_metadata(robot)
    return robot
