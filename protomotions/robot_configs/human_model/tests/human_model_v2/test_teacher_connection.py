import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from protomotions.envs.control.walking_quality import StepTracker, walking_envelope, posture_angles, evaluate_speed_sweep
from protomotions.robot_configs.human_model.human_model_v2.retarget import forward
from protomotions.robot_configs.human_model.common.paths import PACKAGE_ROOT,PROTOMOTIONS_ROOT
from protomotions.components.pose_lib import extract_kinematic_info, fk_batch_mjcf_with_velocities
from protomotions.utils.rotations import quaternion_to_matrix


def test_cadence_counts_both_feet_and_partial_reset():
    t=StepTracker(2,'cpu',.01)
    pos=torch.zeros(2,2,2);direction=torch.tensor([[1.,0.],[1.,0.]])
    for i in range(401):
        phase=i%100
        contact=torch.tensor([[phase<60,phase>=50 or phase<10]]).expand(2,-1)
        pos[:,0,0]=(i//100)*1.2;pos[:,1,0]=((i+50)//100)*1.2-.6
        valid=t.update(contact,pos,direction)
    assert valid.all();torch.testing.assert_close(t.cadence,torch.full((2,),120.),atol=.01,rtol=0)
    assert torch.allclose(t.step_length,torch.full((2,),.6),atol=1e-5)
    t.reset(torch.tensor([0]));assert t.events[0]==0 and t.events[1]>0
    for _ in range(201):t.update(torch.ones(2,2,dtype=torch.bool),pos,direction)
    assert not t.update(torch.ones(2,2,dtype=torch.bool),pos,direction).any()


def test_contact_chatter_and_hops_do_not_become_walk_steps():
    t=StepTracker(1,'cpu',.01);p=torch.zeros(1,2,2);d=torch.tensor([[1.,0.]])
    for i in range(400):t.update(torch.full((1,2),i%2==0),p,d)
    assert t.events.item()==0
    for i in range(400):t.update(torch.full((1,2),i%50<25),p,d)
    assert t.events.item()==0


def test_posture_trunk_and_head_pitch():
    from protomotions.utils.rotations import quat_from_angle_axis
    pos=torch.tensor([[[0.,0.,1.],[.15,0.,1.4],[.15,0.,1.7]]])
    rot=torch.tensor([[[0.,0.,0.,1.]]]).expand(1,3,4).clone()
    rot[:,2]=quat_from_angle_axis(torch.tensor([.4]),torch.tensor([[0.,1.,0.]]),w_last=True)
    trunk,head=posture_angles(rot,pos,0,1,2)
    assert trunk.item()>20 and head.item()>20
    upright=pos.clone();upright[:,:,0]=0
    assert posture_angles(rot,upright,0,1,2)[0].item()==0


def test_missing_speed_coverage_cannot_pass():
    assert evaluate_speed_sweep([])['status']=='candidate_failed_gate'
    assert walking_envelope(torch.tensor(1.5))[0]>walking_envelope(torch.tensor(.5))[0]
    assert walking_envelope(torch.tensor(1.5))[1]>walking_envelope(torch.tensor(.5))[1]


def test_trace_without_post_warmup_samples_fails_closed():
    from protomotions.envs.control.walking_quality import evaluate_walking_trace
    empty=torch.empty(0)
    report=evaluate_walking_trace(empty,empty,empty,empty,empty,empty,[],1/30)
    assert report['status']=='candidate_failed_gate'
    assert evaluate_speed_sweep([report])['status']=='candidate_failed_gate'


def test_retarget_forward_matches_canonical_fk_and_backprop():
    ki=extract_kinematic_info(str(PACKAGE_ROOT/'human_model_v2/assets/human_model_v2.xml'))
    torch.manual_seed(4)
    q=(torch.rand(3,59)*.05).requires_grad_();root=torch.zeros(3,3);rot=torch.eye(3).expand(3,3,3)
    pos,rotation,_=forward(ki,q,root,rot)
    qpos=torch.cat((root,torch.tensor([[1.,0.,0.,0.]]).expand(3,-1),q.detach()),-1)
    state=fk_batch_mjcf_with_velocities(ki,qpos,compute_velocities=False)
    torch.testing.assert_close(pos,state.rigid_body_pos)
    torch.testing.assert_close(rotation,quaternion_to_matrix(state.rigid_body_rot,w_last=True),atol=1e-6,rtol=1e-5)
    pos.square().sum().backward();assert torch.isfinite(q.grad).all() and q.grad.abs().max()>0


def test_teacher_factory_action_contract_and_reference_guard(tmp_path):
    from protomotions.robot_configs.factory import robot_config
    from protomotions.robot_configs.base import ControlType
    spec=importlib.util.spec_from_file_location('v2_steering',PROTOMOTIONS_ROOT/'examples/experiments/steering/human_model_v2.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    robot=robot_config('human_model_v2');assert robot.number_of_actions==59
    args=SimpleNamespace(simulator='isaaclab',motion_file=str(tmp_path/'old.pt'))
    sim=SimpleNamespace(sim=SimpleNamespace(physx=SimpleNamespace()))
    module.configure_robot_and_simulator(robot,sim,args)
    assert robot.control.control_type==ControlType.PROPORTIONAL
    env=module.env_config(robot,args);assert 'walking_quality' in env.reward_components
    assert sim.sim.fps/sim.sim.decimation==30
    torch.save({'dps':torch.zeros(2,69),'gts':torch.zeros(2,24,3)},args.motion_file)
    with pytest.raises(ValueError,match='retargeted'):module.motion_lib_config(args)
    torch.save({'dps':torch.zeros(2,59),'gts':torch.zeros(2,26,3),'human_model_metadata':{'model_id':'human_model_v2','status':'candidate_failed_gate'}},args.motion_file)
    with pytest.raises(ValueError,match='ordering or asset SHA'):module.motion_lib_config(args)
    import hashlib
    d=torch.load(args.motion_file,weights_only=False);d['human_model_metadata'].update(dof_names=robot.kinematic_info.dof_names,body_names=robot.kinematic_info.body_names,asset_sha256=hashlib.sha256((PACKAGE_ROOT/'human_model_v2/assets/human_model_v2.xml').read_bytes()).hexdigest());torch.save(d,args.motion_file)
    with pytest.raises(ValueError,match='quality gate'):module.motion_lib_config(args)


def test_scalar_motionlib_interpolation_avoids_smpl_expmap(tmp_path):
    from protomotions.components.motion_lib import MotionLib,MotionLibConfig
    q=torch.zeros(2,59);q[1]=.2
    pack=dict(gts=torch.zeros(2,26,3),grs=torch.tensor([0.,0.,0.,1.]).expand(2,26,4).clone(),
        gvs=torch.zeros(2,26,3),gavs=torch.zeros(2,26,3),dps=q,dvs=torch.zeros(2,59),
        contacts=torch.ones(2,26,dtype=torch.bool),length_starts=torch.tensor([0]),motion_num_frames=torch.tensor([2]),
        motion_dt=torch.tensor([1.]),motion_lengths=torch.tensor([1.]),motion_weights=torch.tensor([1.]),motion_files=('test',))
    path=tmp_path/'hinges.pt';torch.save(pack,path)
    lib=MotionLib(MotionLibConfig(motion_file=str(path)))
    state=lib.get_motion_state(torch.tensor([0]),torch.tensor([.5]))
    torch.testing.assert_close(state.dof_pos,torch.full((1,59),.1))


def test_speed_sweep_rejects_cadence_only_speedup():
    reports=[]
    for speed in (.5,.75,1.,1.25,1.5):
        reports.append(dict(status='passed',metrics=dict(speed_command_mps=speed,
            cadence_median_spm=60*speed/.4,step_length_median_m=.4)))
    result=evaluate_speed_sweep(reports)
    assert result['checks']['coverage']
    assert not result['checks']['step_length_increases']
    assert not result['checks']['cadence_not_only_strategy']
