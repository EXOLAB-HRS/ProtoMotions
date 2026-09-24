"""v3.1 must restore physical adjacency without changing the old plant."""
from pathlib import Path
import pytest


def test_collision_asset_preserves_plant_and_restores_only_neighbors():
    Usd = pytest.importorskip('pxr.Usd')
    from protomotions.robot_configs.human_model.registry import get_model
    from protomotions.robot_configs.human_model.human_model_v3_1.collision import inspect_stage, validate_collision_asset
    old = get_model('human_model_v3').asset_path('usd')
    new = get_model('human_model_v3.1').asset_path('usd')
    report = validate_collision_asset(old, new)
    assert len(report['restored_pairs']) == 19
    assert report['collider_count'] == 24
    info = inspect_stage(Usd.Stage.Open(str(new)))
    short = {tuple(sorted(p.rsplit('/',1)[-1] for p in pair)) for pair in info['excluded']}
    for pair in [('Head','Neck'), ('Chest','Spine'), ('L_Ankle','L_Knee')]:
        assert tuple(sorted(pair)) in short
    for pair in [('L_Hand','Chest'), ('L_Hip','R_Hip'), ('L_Ankle','R_Ankle')]:
        assert tuple(sorted(pair)) not in short
    original_smpl = Path(old).resolve().parents[4] / 'data/assets/usd/smpl_humanoid.usda'
    legacy = inspect_stage(Usd.Stage.Open(str(original_smpl)))
    assert info['excluded'] == legacy['excluded'] | legacy['adjacent']


def test_v31_reuses_v3_controller_and_forces(monkeypatch):
    monkeypatch.delenv('PROTOMOTIONS_HUMAN_MODEL', raising=False)
    from protomotions.robot_configs.factory import robot_config
    a, b = robot_config('human_model_v3'), robot_config('human_model_v3.1')
    assert a.human_model_profile == b.human_model_profile == 'human_model_v3'
    assert a.human_model_parameters == b.human_model_parameters
    assert a.control == b.control
    assert a.kinematic_info.dof_names == b.kinematic_info.dof_names
    assert b.asset.self_collisions
    assert b.human_model_collision_profile == 'human_model_v3.1'
    assert a.human_model_collision_profile is None


def test_backend_selects_revision_without_changing_forces(monkeypatch):
    pytest.importorskip('pxr.Usd')
    import torch
    from types import SimpleNamespace
    from protomotions.robot_configs.factory import robot_config
    from protomotions.robot_configs.human_model.common.integration import prepare_simulator
    from protomotions.robot_configs.human_model.registry import get_model
    monkeypatch.delenv('PROTOMOTIONS_HUMAN_MODEL', raising=False)
    sim = SimpleNamespace(_target_='protomotions.simulator.isaaclab.simulator.IsaacLabSimulator', num_envs=1)
    a, am = prepare_simulator(robot_config('human_model_v3'), sim, 'cpu')
    b, bm = prepare_simulator(robot_config('human_model_v3.1'), sim, 'cpu')
    assert (Path(b.asset.asset_root)/b.asset.usd_asset_file_name).resolve() == get_model('human_model_v3.1').asset_path('usd').resolve()
    assert (Path(a.asset.asset_root)/a.asset.usd_asset_file_name).resolve() == get_model('human_model_v3').asset_path('usd').resolve()
    assert a.control == b.control
    for field in ('lower', 'upper', 'backend_limit', 'negative', 'positive'):
        assert torch.equal(getattr(am, field), getattr(bm, field))


def test_broad_exclusion_is_rejected(tmp_path):
    Usd = pytest.importorskip('pxr.Usd')
    from pxr import UsdPhysics
    from protomotions.robot_configs.human_model.registry import get_model
    from protomotions.robot_configs.human_model.human_model_v3_1.collision import validate_collision_asset
    path = tmp_path/'bad.usda'
    s = Usd.Stage.CreateNew(str(path))
    s.GetRootLayer().subLayerPaths = [str(get_model('human_model_v3.1').asset_path('usd').resolve())]
    p = s.GetPrimAtPath('/smpl_humanoid/bodies/L_Hand')
    UsdPhysics.FilteredPairsAPI.Apply(p).GetFilteredPairsRel().AddTarget('/smpl_humanoid/bodies/Chest')
    s.GetRootLayer().Save()
    with pytest.raises(ValueError, match='over-broad'):
        validate_collision_asset(get_model('human_model_v3').asset_path('usd'), path)
