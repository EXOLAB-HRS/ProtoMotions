import importlib.util
from pathlib import Path


def _module():
    path = (
        Path(__file__).parents[2]
        / "examples/experiments/steering/mlp_human_model_v3_a7_followups.py"
    )
    spec = importlib.util.spec_from_file_location("a7_followups_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_discriminator_batch_does_not_exceed_ppo_batch():
    module = _module()
    assert module.discriminator_batch_size(16384) == 4096
    assert module.discriminator_batch_size(4096) == 4096
    assert module.discriminator_batch_size(512) == 512
