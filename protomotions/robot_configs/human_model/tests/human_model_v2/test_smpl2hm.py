import hashlib
import torch
from protomotions.components.motion_lib import MotionLibConfig
from protomotions.components.pose_lib import extract_kinematic_info
from protomotions.robot_configs.human_model.common.paths import PACKAGE_ROOT
from protomotions.robot_configs.human_model.human_model_v2.retarget import forward
from protomotions.robot_configs.human_model.human_model_v2.retarget_metrics import derivative, intervals, contact_metrics
from protomotions.robot_configs.human_model.human_model_v2.motion_lib import HumanModelMotionLib
from protomotions.utils.rotations import matrix_to_quaternion, quaternion_to_matrix


def test_contact_runs_exclude_airborne_values_and_keep_endpoints():
    assert list(intervals(torch.tensor([1,1,0,1,1],dtype=torch.bool)))==[(0,2),(3,5)]
    points=torch.zeros(10,4,3);points[2:8,:,0]=100
    rows=contact_metrics(points,[(0,0,2,0),(2,8,10,0)],.1)
    assert all(r['slip_p95_mps']==0 and r['drift_max_m']==0 for r in rows)
    assert contact_metrics(points,[],.1)[0]['slip_p95_mps'] is None


def test_derivative_units_and_endpoints():
    t=torch.arange(10)*.02
    torch.testing.assert_close(derivative(3*t,.02),torch.full((10,),3.),atol=1e-5,rtol=1e-5)


def test_hinge_interpolation_reconstructs_fk_velocity_and_contact(tmp_path):
    asset=PACKAGE_ROOT/'human_model_v2/assets/human_model_v2.xml'
    ki=extract_kinematic_info(str(asset))
    q=torch.zeros(3,59);q[:,0]=torch.tensor([-.1,-.3,-.5])
    root=torch.tensor([[0.,0.,1.],[.1,0.,1.],[.2,0.,1.]])
    pos,rot,_=forward(ki,q,root,torch.eye(3).expand(3,3,3))
    contacts=torch.zeros(3,26,dtype=torch.bool);contacts[1:]=True
    pack=dict(gts=pos,grs=matrix_to_quaternion(rot,w_last=True),gvs=torch.zeros_like(pos),gavs=torch.zeros_like(pos),
        dps=q,dvs=torch.zeros_like(q),contacts=contacts,motion_num_frames=torch.tensor([3]),
        length_starts=torch.tensor([0]),motion_dt=torch.tensor([.1]),motion_lengths=torch.tensor([.2]),
        motion_weights=torch.tensor([1.]),motion_files=('synthetic',),human_model_metadata=dict(model_id='human_model_v2',
            asset_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),body_names=ki.body_names,dof_names=ki.dof_names))
    path=tmp_path/'v2.pt';torch.save(pack,path)
    lib=HumanModelMotionLib(MotionLibConfig(motion_file=str(path)))
    state=lib.get_motion_state(torch.zeros(3,dtype=torch.long),torch.tensor([0.,.05,.2]))
    expected=forward(ki,state.dof_pos,state.rigid_body_pos[:,0],quaternion_to_matrix(state.rigid_body_rot[:,0],w_last=True))[0]
    torch.testing.assert_close(state.rigid_body_pos,expected)
    torch.testing.assert_close(state.dof_vel[:,0],torch.full((3,),-2.),atol=1e-4,rtol=1e-4)
    torch.testing.assert_close(state.rigid_body_vel[:,0,0],torch.ones(3),atol=1e-4,rtol=1e-4)
    assert not state.rigid_body_contacts[1].any()
    assert state.rigid_body_contacts[2].all()
    assert torch.isfinite(state.rigid_body_ang_vel).all()


def test_normalization_recovers_fixed_skeleton_without_changing_timing():
    from protomotions.robot_configs.human_model.human_model_v2.retarget_metrics import normalize_skeleton
    ki=extract_kinematic_info(str(PACKAGE_ROOT/'human_model_v1/assets/smpl_humanoid.xml'))
    q=torch.zeros(6,69);root=torch.zeros(6,3);root[:,2]=1;root[:,0]=torch.arange(6)*.05
    pos,_,_=forward(ki,q,root,torch.eye(3).expand(6,3,3))
    normalized,meta=normalize_skeleton(pos*1.2,ki)
    torch.testing.assert_close(normalized,pos,atol=1e-6,rtol=1e-5)
    assert abs(meta['root_translation_scale']-1/1.2)<1e-5


def test_turning_contact_point_does_not_confuse_body_origin_with_slip():
    from protomotions.robot_configs.human_model.human_model_v2.retarget_metrics import material_points
    angle=torch.linspace(0,1,40);r=torch.eye(3).expand(40,1,3,3).clone()
    r[:,0,0,0]=angle.cos();r[:,0,0,1]=-angle.sin();r[:,0,1,0]=angle.sin();r[:,0,1,1]=angle.cos()
    local=torch.tensor([.1,0.,0.]).expand(40,1,3)
    pos=-torch.einsum('tfij,tfj->tfi',r,local)
    point=material_points(pos,r,[0],local)
    assert pos[:,0,:2].std()>0
    torch.testing.assert_close(point,torch.zeros_like(point))


def test_free_base_gravity_wrench_is_not_a_joint_torque():
    from protomotions.robot_configs.human_model.human_model_v2.validation import joint_generalized_forces
    forces=torch.tensor([[0.,0.,621.,0.,0.,0.,2.,3.,4.]])
    order=torch.tensor([2,0,1])
    torch.testing.assert_close(joint_generalized_forces(forces,order,floating_base=True),torch.tensor([[4.,2.,3.]]))
    torch.testing.assert_close(joint_generalized_forces(forces[:,6:],order,floating_base=False),torch.tensor([[4.,2.,3.]]))


def test_subset_preserves_model_contract_and_reorders_clip_evidence(tmp_path):
    from scripts.subset_motion_lib import subset_motion_lib
    source = tmp_path / 'source.pt'
    target = tmp_path / 'selected.pt'
    metadata = dict(model_id='human_model_v2', asset_sha256='fixed-asset',
                    dof_names=['joint'], body_names=['body'],
                    status='candidate_requires_confirmation',
                    motions=[dict(motion_id=i, status='screening_passed') for i in range(3)])
    torch.save(dict(gts=torch.arange(18).reshape(6,1,3),
                    motion_lengths=torch.tensor([.1,.1,.1]),
                    motion_num_frames=torch.tensor([2,2,2]),
                    length_starts=torch.tensor([0,2,4]),
                    motion_dt=torch.tensor([.1,.1,.1]),
                    motion_weights=torch.ones(3), motion_files=('a','b','c'),
                    human_model_metadata=metadata), source)
    subset_motion_lib(str(source), str(target), indices=[2,0])
    selected = torch.load(target, weights_only=False)
    meta = selected['human_model_metadata']
    assert meta['status'] == metadata['status']
    assert meta['asset_sha256'] == metadata['asset_sha256']
    assert meta['dof_names'] == metadata['dof_names']
    assert [m['motion_id'] for m in meta['motions']] == [2,0]
    assert meta['subset_parent']['sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert selected['motion_files'] == ('c','a')
    assert selected['length_starts'].tolist() == [0,2]
    assert selected['gts'][:,0,0].tolist() == [12,15,0,3]


def test_subject_shape_import_matches_independent_smplh(monkeypatch):
    """e03 candidate: licensed models supplied by env; compare against SMPL-X LBS."""
    import os
    import json
    import numpy as np
    import pytest
    from pathlib import Path
    model_root = os.environ.get('SMPLH_MODEL_DIR')
    if not model_root:
        pytest.skip('Set SMPLH_MODEL_DIR to licensed male/female/neutral models')
    from importlib.metadata import distribution
    # pytest prepends robot_configs, which also contains a file named smplx.py.
    monkeypatch.syspath_prepend(str(distribution('smplx').locate_file('')))
    smplx = pytest.importorskip('smplx')
    repository = Path(__file__).resolve().parents[5]
    monkeypatch.syspath_prepend(str(repository/'data/scripts'))
    from data.scripts.convert_amass_to_proto import smplh_subject_kinematics, convert_amass_to_motion
    from data.smpl.smpl_joint_names import SMPL_BONE_ORDER_NAMES, SMPL_MUJOCO_NAMES
    from protomotions.robot_configs.human_model.human_model_v2.retarget_metrics import normalize_skeleton
    ki = extract_kinematic_info(str(repository/'protomotions/data/assets/mjcf/smpl_humanoid.xml'))
    rows = []
    for gender in ('male', 'female', 'neutral'):
        beta = np.linspace(-.3, .4, 16)
        shaped, offset, meta = smplh_subject_kinematics(model_root, beta, gender, ki)
        model = np.load(Path(model_root)/gender/'model.npz', allow_pickle=False)
        poses = np.random.default_rng(22).normal(0, .12, (8,156))
        trans = np.random.default_rng(2).normal(0, .2, (8,3))
        tensor = lambda a: torch.tensor(a, dtype=torch.float32)
        from smplx.lbs import lbs
        parents = torch.tensor(model['kintree_table'][0].astype(np.int64));parents[0]=-1
        with torch.no_grad():
            _, ref = lbs(tensor(beta)[None].expand(8,-1), tensor(poses),
                         tensor(model['v_template']), tensor(model['shapedirs']),
                         tensor(model['posedirs']).reshape(-1,459).T, tensor(model['J_regressor']),
                         parents, tensor(model['weights']))
            ref = ref + tensor(trans)[:,None]
        model.close()
        motion, fps = convert_amass_to_motion(poses, trans+offset, 30, 30, 'smpl',
            SMPL_BONE_ORDER_NAMES, SMPL_MUJOCO_NAMES, shaped, torch.device('cpu'), torch.float32)
        idx = [SMPL_BONE_ORDER_NAMES.index(n) for n in ki.body_names]
        idx = [25 if i==22 else 40 if i==23 else i for i in idx]
        error = ((motion.rigid_body_pos-motion.rigid_body_pos[:,:1])-(ref[:,idx]-ref[:,:1])).norm(dim=-1).max()
        assert error < 1e-5
        torch.testing.assert_close(motion.rigid_body_pos[:,0,:2], ref[:,0,:2], atol=1e-6, rtol=1e-6)
        normalized, scale = normalize_skeleton(motion.rigid_body_pos, ki)
        for b in range(1,24):
            measured = (normalized[:,b]-normalized[:,ki.parent_indices[b]]).norm(dim=-1)
            torch.testing.assert_close(measured, ki.local_pos[b].norm().expand_as(measured), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(normalized[:,0,:2]-normalized[:1,0,:2],
            (motion.rigid_body_pos[:,0,:2]-motion.rigid_body_pos[:1,0,:2])*scale['root_translation_scale'])
        assert fps == 30 and len(normalized)==8
        with pytest.raises(ValueError, match='coefficient'):
            smplh_subject_kinematics(model_root, beta[:10], gender, ki)
        rows.append(dict(gender=gender, independent_joint_error_m=float(error), model_sha=meta['body_model_sha256'], root_scale=scale['root_translation_scale']))
    if os.environ.get('SMPLH_VALIDATION_OUTPUT'):
        Path(os.environ['SMPLH_VALIDATION_OUTPUT']).write_text(json.dumps(dict(status='passed', reference='smplx.lbs independent full LBS; randomized rotations/nonzero betas', results=rows),indent=2)+'\n')


def test_upper_smoothing_preserves_legs_rom_and_low_frequency_motion():
    from protomotions.robot_configs.human_model.human_model_v2.retarget_metrics import smooth_upper_hinges
    for fps in (30,50):
        t=torch.arange(4*fps)/fps
        slow=.5*torch.sin(2*torch.pi*t)
        noise=.12*torch.cos(2*torch.pi*10*t)
        leg=.3*torch.sin(2*torch.pi*2*t)
        q=torch.stack((leg,slow+noise,slow-noise),-1)
        out=smooth_upper_hinges(q,['L_Knee_y','Chest_x','L_Thorax_x'],1/fps,3.)
        assert torch.equal(q[:,0],out[:,0])
        assert torch.equal(q[[0,-1]],out[[0,-1]])
        assert out[:,1].min()>=q[:,1].min()-1e-6 and out[:,1].max()<=q[:,1].max()+1e-6
        interior=slice(fps,3*fps)
        assert (out[interior,1]-slow[interior]).square().mean().sqrt()<.035
        before=torch.diff(q[:,1],n=3).abs().max()
        after=torch.diff(out[:,1],n=3).abs().max()
        assert after<.3*before
        assert torch.diff(out[:,2],n=3).abs().max()<.3*torch.diff(q[:,2],n=3).abs().max()


def test_discrete_jerk_matches_cubic_translation_and_rotation_in_si_units():
    from protomotions.robot_configs.human_model.human_model_v2.retarget_metrics import temporal_jerk_metrics
    from scipy.spatial.transform import Rotation
    import numpy as np
    for dt in (.02,.033333333333):
        t=torch.arange(12,dtype=torch.float64)*dt
        names=['Pelvis','Chest','L_Shoulder','R_Shoulder','L_Elbow','R_Elbow','L_Hand','R_Hand']
        p=torch.zeros(len(t),8,3,dtype=torch.float64);p[:,1:,0]=t[:,None]**3
        r=torch.eye(3,dtype=torch.float64).repeat(len(t),8,1,1)
        r[:,1:]=torch.from_numpy(Rotation.from_rotvec(np.stack((np.zeros(len(t)),np.zeros(len(t)),t.numpy()**3),-1)).as_matrix())[:,None]
        q=torch.stack((t**3,t*0),-1)
        m=temporal_jerk_metrics(q,p,r,['Chest_x','L_Knee_y'],names,dt)
        for key in ('upper_hinge_jerk_rad_s3','upper_relative_position_jerk_m_s3','chest_shoulders_relative_angular_jerk_rad_s3'):
            assert abs(m[key]['max']-6)<1e-8
        assert m['root_position_jerk_m_s3']['max']==0


def test_upper_trust_projection_bounds_geometry_without_changing_legs():
    from protomotions.robot_configs.human_model.human_model_v2.retarget import project_upper_correction
    ki=extract_kinematic_info(str(PACKAGE_ROOT/'human_model_v2/assets/human_model_v2.xml'))
    q=torch.zeros(30,59);ref=q.clone();q[:,ki.dof_names.index('Chest_y')]=.2*torch.sin(torch.linspace(0,torch.pi,30));q[[0,-1]]=0
    root=torch.zeros(30,3);root[:,2]=1.;rot=torch.eye(3).expand(30,3,3)
    projected,alpha=project_upper_correction(ki,q,ref,root,rot,.02)
    assert 0<alpha<1 and torch.equal(projected[:,:14],ref[:,:14])
    assert torch.equal(projected[[0,-1]],q[[0,-1]])
    pos=forward(ki,projected,root,rot)[0];original=forward(ki,ref,root,rot)[0]
    assert float((pos-original).norm(dim=-1).max())<=.020001
    torch.testing.assert_close(projected,alpha*q)


def test_source_trunk_stays_in_source_range_with_source_speed_and_jerk_limits():
    from protomotions.robot_configs.human_model.human_model_v2.retarget_metrics import source_trunk_trajectory,TRUNK_SEGMENTS
    ski=extract_kinematic_info(str(PACKAGE_ROOT/'human_model_v1/assets/smpl_humanoid.xml'))
    ki=extract_kinematic_info(str(PACKAGE_ROOT/'human_model_v2/assets/human_model_v2.xml'))
    n=90;dt=1/30;t=torch.arange(n)*dt
    sq=torch.zeros(n,69)
    for name in TRUNK_SEGMENTS:
        sq[:,ski.dof_names.index(name+'_x')]=.025*torch.sin(2*torch.pi*t)+.005*torch.cos(2*torch.pi*9*t)
    sq[:,ski.dof_names.index('L_Thorax_y')]=.7  # source neutral outside target ROM
    root=torch.zeros(n,3);rot=torch.eye(3).expand(n,3,3)
    sr=forward(ski,sq,root,rot)[1];base=torch.zeros(n,59)
    q,report=source_trunk_trajectory(base,sr,ski,ki,dt)
    assert torch.equal(q[:,:14],base[:,:14])
    assert all(x['source_envelope_pass'] for x in report)
    assert all(x['output_speed_max_rad_s']<=x['speed_limit_rad_s']*1.001+1e-5 for x in report)
    assert all(x['output_angular_jerk_max_rad_s3']<=x['jerk_limit_rad_s3']*1.001+1e-3 for x in report)
    assert torch.all(q>=ki.dof_limits_lower-1e-6) and torch.all(q<=ki.dof_limits_upper+1e-6)
    assert abs(next(x for x in report if x['segment']=='L_Thorax')['neutral_offset_rad'][1]+.7)<1e-5
