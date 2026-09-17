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
