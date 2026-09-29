import importlib
import math


def test_a11_is_scratch_omni_without_stop_or_head_reward():
    module = importlib.import_module(
        "examples.experiments.steering.mlp_human_model_v3_a11_proto1"
    )
    assert module.EXPERIMENT_ID == "a11-proto1"
    assert module.STOP_PROBABILITY == 0.0
    assert module.HEAD_REWARD_WEIGHT == 0.0


def test_a12_uses_reference_head_envelope_and_stronger_reward():
    module = importlib.import_module(
        "examples.experiments.steering.mlp_human_model_v3_a12_scratch_head"
    )
    assert module.EXPERIMENT_ID == "a12"
    assert module.HEAD_REWARD_WEIGHT == 0.15
    assert math.isclose(module.HEAD_TILT_LIMIT_RAD, math.radians(15.7145))
    assert math.isclose(module.HEAD_RELATIVE_SPEED_LIMIT_RAD_S, math.radians(77.2072))


def test_a13_curriculum_mixtures_preserve_rehearsal_and_expand_by_stage():
    module = importlib.import_module(
        "examples.experiments.steering.mlp_human_model_v3_a13_curriculum"
    )
    assert module.HEAD_REWARD_WEIGHT == 0.15
    assert module.STAGES["b"]["fixed_fraction"] == 0.30
    assert module.STAGES["b"]["stop_probability"] == 0.0
    assert module.STAGES["c"]["fixed_fraction"] == 0.20
    assert module.STAGES["c"]["stop_probability"] > 0
    assert module.STAGES["d"]["turn_fraction"] == 0.60
    assert module.STAGES["d"]["full_heading_for_turn"] is True
    assert module.STAGES["d"]["independent_facing_fraction"] == 1.0
