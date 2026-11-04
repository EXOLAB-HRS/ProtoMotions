"""human_model_v3.4 (2026-11-04, user-approved version on the v3.4 proposal): human_model_v3.3 without the gastrocnemius
passive knee-ankle term (passive_v3_4.py). Everything else (profile, strength, PD gains, assets, collision) is v3.3 = v3.1."""
MODEL = {
    "model_id": "human_model_v3.4", "resource_directory": "human_model_v3_1",
    "profile_id": "human_model_v3", "status": "candidate", "runnable": True,
    "assets": {"usd": "assets/human_model_v3_1.usda",
               "mjcf": "../human_model_v2/assets/human_model_v2.xml"},
    "profile": "../human_model_v3/profiles/healthy_adult_v3.json",
    "kinematic_model_id": "human_model_v2",
}
def passive_exponentials():
    from copy import deepcopy
    from .passive_v3_4 import PASSIVE_EXPONENTIALS
    return deepcopy(PASSIVE_EXPONENTIALS)


def robot_config():
    from ..human_model_v3_1.model_config import robot_config as v31_config
    from ..common.integration import apply_metadata
    robot = v31_config()
    robot.human_model_parameters = dict(robot.human_model_parameters, passive_exponentials=passive_exponentials())
    apply_metadata(robot)
    return robot
