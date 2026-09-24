"""Collision-only v3 revision; single-condition screening passed, candidate."""

MODEL = {
    "model_id": "human_model_v3.1", "resource_directory": "human_model_v3_1",
    "profile_id": "human_model_v3", "status": "candidate", "runnable": True,
    "assets": {"usd": "assets/human_model_v3_1.usda",
               "mjcf": "../human_model_v2/assets/human_model_v2.xml"},
    "profile": "../human_model_v3/profiles/healthy_adult_v3.json",
    "kinematic_model_id": "human_model_v2",
}


def robot_config():
    from ..human_model_v3.model_config import robot_config as v3_config
    robot = v3_config()
    robot.human_model_collision_profile = MODEL["model_id"]
    robot.asset.self_collisions = True
    return robot
