"""human_model_v3.3 (2026-11-02, user: "수동강성은 문헌값으로 낮춰서 human_model_v3.3으로"): human_model_v3.1 (collision
revision, Neck/Head Kd x2) with the passive leg exponentials replaced by the set fitted to the digitized Silder 2007 curves
(passive_v3_3.py). Everything else (profile, strength, PD gains, assets) is v3.1."""
MODEL = {
    "model_id": "human_model_v3.3", "resource_directory": "human_model_v3_1",
    "profile_id": "human_model_v3", "status": "candidate", "runnable": True,
    "assets": {"usd": "assets/human_model_v3_1.usda",
               "mjcf": "../human_model_v2/assets/human_model_v2.xml"},
    "profile": "../human_model_v3/profiles/healthy_adult_v3.json",
    "kinematic_model_id": "human_model_v2",
}
def passive_exponentials():
    from copy import deepcopy
    from .passive_v3_3 import PASSIVE_EXPONENTIALS
    return deepcopy(PASSIVE_EXPONENTIALS)


def robot_config():
    from ..human_model_v3_1.model_config import robot_config as v31_config
    from ..common.integration import apply_metadata
    robot = v31_config()
    robot.human_model_parameters = dict(robot.human_model_parameters, passive_exponentials=passive_exponentials())
    apply_metadata(robot)
    return robot
