"""human_model_v3.5 (2026-11-07, user-approved version: "플랜트 질량, 관성모멘트가 ... 업뎃필요하면 버젼 새버젼 해서 업뎃해줘"):
human_model_v3.4 with human segment mass properties (de Leva 1996 sex mean; build_bsp_v3_5.py, bsp_v3_5.json). Only body
masses, centres of mass and inertias differ; total mass 63.31 kg, geometry, colliders, joints, ROM, strength, PD gains and
the v3.4 passive terms are unchanged. Selected at runtime by human_model_parameters["segment_inertia"] = "human_model_v3.5"
(common/integration.py swaps the v3.1 collision USD for assets/human_model_v3_5.usda, which sublayers it)."""
MODEL = {
    "model_id": "human_model_v3.5", "resource_directory": "human_model_v3_5",
    "profile_id": "human_model_v3", "status": "candidate", "runnable": True,
    "assets": {"usd": "assets/human_model_v3_5.usda", "mjcf": "assets/human_model_v3_5.xml"},
    "profile": "../human_model_v3/profiles/healthy_adult_v3.json",
    "kinematic_model_id": "human_model_v2",
}


def passive_exponentials():
    from ..human_model_v3_4.model_config import passive_exponentials as pe34
    return pe34()


def robot_config():
    from ..human_model_v3_4.model_config import robot_config as v34_config
    from ..common.integration import apply_metadata
    robot = v34_config()
    robot.human_model_parameters = dict(robot.human_model_parameters, segment_inertia=MODEL["model_id"])
    apply_metadata(robot)
    return robot
