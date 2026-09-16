"""Calibration correctness and study-split leakage regression tests."""

import json

import numpy as np
import torch
import pytest

from protomotions.robot_configs.human_model.calibration import fit_silder, screen_passive_fit, screen_isometric_strength
from protomotions.robot_configs.human_model.dynamics import HumanJointModel
from protomotions.robot_configs.human_model.profile import load_profile


def test_rounded_target_capacity_intersection_preserves_quantization():
    from protomotions.robot_configs.human_model.calibration import capacity_interval_from_rounded_targets
    interval=capacity_interval_from_rounded_targets([12.9,25.8],[.2,.4],rounding_step=.1)
    np.testing.assert_allclose(interval,[64.375,64.625])
    np.testing.assert_allclose(capacity_interval_from_rounded_targets([10.,20.1],[.2,.4],rounding_step=.1),[50.125,50.25])
    with pytest.raises(ValueError,match='incompatible'):
        capacity_interval_from_rounded_targets([10,30],[.2,.4],rounding_step=.1)
    for targets,fractions,step in [([],[],.1),([1],[0],.1),([-1],[.2],.1),([1],[.2],0),([float('nan')],[.2],.1)]:
        with pytest.raises(ValueError):
            capacity_interval_from_rounded_targets(targets,fractions,rounding_step=step)


def test_isokinetic_coordinates_preserve_shortening_work_and_zero_torque_motion():
    from protomotions.robot_configs.human_model.calibration import isokinetic_joint_state
    for joint in ('L_Hip_y', 'L_Knee_y', 'L_Ankle_y'):
        for direction in (-1, 1):
            values = np.array([[60., -60., 60.], [0., 0., 0.], [10.*direction, 10.*direction, 0.]])
            q, qd, tau = isokinetic_joint_state(values, joint, direction)
            assert qd[0]*tau[0] > 0  # Shortening must deliver positive joint work.
            assert qd[1]*tau[1] < 0  # Lengthening must absorb joint work.
            assert qd[2] == qd[0]  # Zero force must not erase commanded velocity.
            assert abs(qd[0]) == pytest.approx(np.pi/3)
    with pytest.raises(ValueError):
        isokinetic_joint_state([[30], [0], [10]], 'L_Ankle_y', -1)


def test_source_cohort_requires_explicit_unmixed_counts():
    from protomotions.robot_configs.human_model.calibration import reported_strength_cohort
    assert reported_strength_cohort('14M (29±6yr; 180±4cm; 79±5kg)') == 'male'
    assert reported_strength_cohort('12 F (22yr)') == 'female'
    assert reported_strength_cohort('14M 12F (22yr)') is None
    assert reported_strength_cohort('unknown') is None


def test_source_size_requires_explicit_mean_units_and_rejects_ranges():
    from protomotions.robot_configs.human_model.calibration import reported_strength_size
    assert reported_strength_size('14M (29±6yr; 180±4cm; 79±5kg)') == (79,1.8)
    assert reported_strength_size('8F (23yr; 170cm; 65kg)') == (65,1.7)
    for unknown in ('14M (29yr; 170~180cm; 79kg)', '14M (29yr; 1.8m; 79kg)',
                    '14M (29yr; 180cm; 79kg); 12F (24yr; 170cm; 65kg)', 'no body data'):
        assert reported_strength_size(unknown) is None


def _synthetic_curves(study, slope_scale=1.0):
    profile = load_profile()
    for component in profile['passive_exponentials']:
        component['coeff_per_rad']={name:value*slope_scale for name,value in component['coeff_per_rad'].items()}
    model = HumanJointModel(profile, list(profile['joints']), dtype=torch.float64)
    rng = np.random.default_rng(821)
    q = rng.uniform(model.lower.numpy(), model.upper.numpy(), size=(80, 69))
    torque = model.elastic_torque(torch.tensor(q)).numpy()
    return [dict(id=f'{study}_{joint}', reference=study, joint=joint,
                 status='candidate', measure_type='Passive',
                 q_rad=q.tolist(), torque_nm=torque[:, model.names.index(joint)].tolist())
            for joint in ('L_Hip_y', 'L_Knee_y', 'L_Ankle_y')]


def test_energy_fit_recovers_synthetic_torques_with_nonnegative_amplitudes():
    records = _synthetic_curves('synthetic')
    fit = fit_silder([], selected=records,
                     scales={r['joint']: 100.0 for r in records})
    assert max(r['candidate']['nrmse'] for r in fit['comparison']) < 1e-10
    assert all(c['energy_scale_j'] >= 0 for c in fit['passive_exponentials'])


def test_scaled_energy_fit_recovers_known_softened_curve_without_negative_energy():
    records=_synthetic_curves('synthetic_softened',slope_scale=.75)
    fitted=fit_silder([],selected=records,scales={r['joint']:100. for r in records},slope_scale=.75)
    assert max(r['candidate']['nrmse'] for r in fitted['comparison'])<1e-10
    assert all(c['energy_scale_j']>=0 for c in fitted['passive_exponentials'])
    assert fitted['slope_scale']==.75
    import pytest
    for scale in (0,-1,float('nan')):
        with pytest.raises(ValueError,match='slope scale'):
            fit_silder([],selected=records,slope_scale=scale)


def test_screening_folds_exclude_held_out_study_and_never_read_confirmation(tmp_path):
    configs = tmp_path / 'configs'
    configs.mkdir()
    (tmp_path / 'evaluation' / 'screening').mkdir(parents=True)
    studies = {'cal': 'calibration', 'a': 'screening', 'b': 'screening',
               'c': 'screening', 'unseen': 'confirmation'}
    records = sum((_synthetic_curves(s) for s in ('cal', 'a', 'b', 'c')), [])
    # No angle/torque fields: attempting to process confirmation would fail.
    records.append({'id': 'unseen', 'reference': 'unseen',
                    'status': 'candidate', 'measure_type': 'Passive'})
    (configs / 'curve_selection_registry.json').write_text(
        json.dumps({'records': records, 'sha256': {}}))
    (configs / 'passive_study_split.json').write_text(json.dumps({'study_split': studies}))
    screen_passive_fit(tmp_path)
    result = json.loads((tmp_path / 'evaluation' / 'screening' /
                         'healthy_adult_v1_passive_study_fit_seed_none.json').read_text())
    assert len(result['folds']) == 3
    for fold in result['folds']:
        held_out_ids = {r['id'] for r in records if r['reference'] == fold['held_out_study']}
        assert held_out_ids.isdisjoint(fold['training_curve_ids'])
        assert 'unseen' not in fold['training_curve_ids']
        assert {r['id'] for r in fold['curves']} == held_out_ids
    assert result['candidate_study_macro_nrmse'] < 1e-10


def test_isometric_direction_and_shared_study_holdout(tmp_path):
    configs=tmp_path/'configs'; configs.mkdir()
    (tmp_path/'evaluation'/'screening').mkdir(parents=True)
    profile=load_profile();names=list(profile['joints'])
    model=HumanJointModel(profile,names,dtype=torch.float64,features=('strength',))
    q=torch.zeros((3,69),dtype=torch.float64)
    i=names.index('L_Ankle_y')
    q[:,i]=torch.tensor([-.1,0,.1])
    q[:,names.index('L_Knee_y')]=np.deg2rad(50)
    negative,positive,_=model.strength_caps(q,torch.zeros_like(q))
    records=[]
    for direction,torque in [('plantarflexion',positive[:,i]),('dorsiflexion',-negative[:,i])]:
        records.append(dict(id=direction,reference='screen',status='candidate',measure_type='Active',
            object='Ankle '+direction,joint='L_Ankle_y',q_rad=q.tolist(),torque_nm=torque.tolist(),
            secondary_angle_deg=[50]*3,subject='synthetic',doi=None,metadata={},evidence_kind='synthetic'))
    records.append(dict(id='unseen',reference='holdout',status='candidate',measure_type='Active'))
    (configs/'curve_selection_registry.json').write_text(json.dumps({'records':records,'sha256':{}}))
    (configs/'passive_study_split.json').write_text(json.dumps({'study_split':{'screen':'screening','holdout':'confirmation'}}))
    screen_isometric_strength(tmp_path)
    result=json.loads((tmp_path/'evaluation'/'screening'/'healthy_adult_v1_isometric_study_comparison_seed_none.json').read_text())
    assert len(result['curves'])==2
    for r in result['curves']:
        assert r['candidate']['nrmse']<1e-12
        assert r['source_knee_protocol_match']
        assert r['comparison_target_gate']['status']=='insufficient_evidence'
