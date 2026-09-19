"""Candidate v3 scope, force accounting and independent gain contracts."""
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from protomotions.robot_configs.human_model.registry import get_model
from protomotions.robot_configs.human_model.profile import load_profile
from protomotions.robot_configs.human_model.dynamics import HumanJointModel
from protomotions.robot_configs.human_model.human_model_v3.model_config import robot_config,configure_pd,configure_strength_profile


def test_v3_reuses_kinematics_and_does_not_change_v2():
    assert get_model('human_model_v3').asset_path('usd').resolve()==get_model('human_model_v2').asset_path('usd').resolve()
    a=load_profile('human_model_v2');b=load_profile('human_model_v3')
    assert a['joints']['Torso_y']['active_nm']==[284,121.75]
    assert len(b['joints'])==59
    for name in a['joints']:
        assert a['joints'][name]['rom_deg']==b['joints'][name]['rom_deg']
    assert b['joints']['R_Thorax_x']['active_nm']==[20,20]


def test_strength_scenarios_leave_pd_fixed_and_scale_force_caps():
    robot=robot_config();sim=SimpleNamespace(sim=SimpleNamespace())
    configure_pd(robot,sim)
    gains={n:(i.stiffness,i.damping) for n,i in robot.control.control_info.items()}
    models=[]
    for label in ('low','reference','high'):
        configure_strength_profile(robot,label)
        assert gains=={n:(i.stiffness,i.damping) for n,i in robot.control.control_info.items()}
        models.append(HumanJointModel(load_profile('human_model_v3'),robot.kinematic_info.dof_names,**robot.human_model_parameters))
    q=torch.zeros(3,59);v=torch.zeros_like(q)
    for direction in (0,1):
        torch.testing.assert_close(models[0].strength_caps(q,v)[direction],models[1].strength_caps(q,v)[direction]*.8)
        torch.testing.assert_close(models[2].strength_caps(q,v)[direction],models[1].strength_caps(q,v)[direction]*1.2)


def test_v3_random_force_passivity_and_backend_budget():
    robot=robot_config()
    model=HumanJointModel(load_profile('human_model_v3'),robot.kinematic_info.dof_names,dtype=torch.float64,**robot.human_model_parameters)
    gen=torch.Generator().manual_seed(2941)
    q=model.lower+(model.upper-model.lower)*torch.rand(1024,59,generator=gen,dtype=torch.float64)
    q.requires_grad_();v=torch.randn(1024,59,generator=gen,dtype=q.dtype)*10
    total,active,passive=model.torques(torch.ones_like(q)*1e5,q,v)
    n,p,_=model.strength_caps(q,v)
    assert torch.isfinite(total).all()
    assert (active<=p+1e-8).all() and (active>=-n-1e-8).all()
    assert (total.abs()<=model.backend_limit+1e-8).all()
    grad=torch.autograd.grad(model.potential_energy(q).sum(),q)[0]
    torch.testing.assert_close(model.elastic_torque(q),-grad)
    assert (model.damping_torque(v)*v<=0).all()


def test_audit_selected_direction_and_continuous_saturation():
    from protomotions.robot_configs.human_model.common.strength_audit import summarize_trace
    trace=torch.zeros(3,8,1,2)
    trace[:,2,0,:]=torch.tensor([[3.,0.],[3.,0.],[-1.,0.]])
    trace[:,5:7]=2
    trace[:,1,0,0]=torch.tensor([3.,0.,-3.])
    record=dict(trace=trace,fields=['q','qd','requested','active','passive','negative','positive','outside'],
                names=['L_Test','R_Test'],physics_fps=10,checkpoint='synthetic')
    profile={'active_strength_model':{'directions':{'positive':dict(joint='Test',sign=1,
        clinical_angle_sign=1,angle_domain_rad=[-1,1],concentric_limit_rad_s=2,eccentric_limit_rad_s=2)}}}
    result=summarize_trace(record,profile)['joints']['L_Test']
    assert result['saturation_fraction']==2/3
    assert result['longest_saturation_s']==.2
    d=result['direction_domains']['positive']
    assert d['either_fraction']==2/3
    assert d['selected_direction_samples']==2
    assert d['selected_direction_outside_fraction']==.5


def test_mvc_target_size_and_direction_specific_provenance():
    import copy
    profile=load_profile('human_model_v3');robot=robot_config()
    direct=HumanJointModel(profile,robot.kinematic_info.dof_names,**robot.human_model_parameters)
    large=copy.deepcopy(profile);large['strength_reference_size']=[126.62,1.754]
    scaled=HumanJointModel(large,robot.kinematic_info.dof_names,**robot.human_model_parameters)
    q=torch.zeros(4,59);v=torch.ones_like(q)
    a=direct.strength_caps(q,v);b=scaled.strength_caps(q,v)
    indices=[i for i,n in enumerate(direct.names) if n.endswith(('Hip_y','Knee_y','Ankle_y'))]
    for direction in (0,1):torch.testing.assert_close(b[direction][:,indices],2*a[direction][:,indices])
    decisions=json.loads((Path(__file__).parents[2]/'human_model_v3/profiles/joint_decisions.json').read_text())
    assert set(decisions)==set(direct.names)
    assert all('uncertainty' in d['normalization'] for d in decisions.values())
    assert profile['joints']['L_Hip_x']['active_nm'][0]==170.5
    assert abs(profile['joints']['L_Hip_x']['active_nm'][1]-126.6*63.31/71.8)<1e-9
    assert profile['joints']['R_Hip_x']['active_nm']==profile['joints']['L_Hip_x']['active_nm'][::-1]


def test_gait_cycle_comparison_uses_net_torque_and_contact_onsets():
    import math
    from protomotions.robot_configs.human_model.common.strength_audit import gait_cycle_moments
    fps=100;time=torch.arange(800)/fps
    values=torch.zeros(800,8,1,6)
    values[:,3,0,:]=torch.sin(time[:,None]*2*math.pi)*2
    values[:,4,0,:]=torch.sin(time[:,None]*2*math.pi)*3
    contact=torch.zeros(800,1,2,3)
    contact[:,:,:,2]=((time%1)<.6)[:,None,None]*100.
    record=dict(trace=values,fields=['q','qd','requested','active','passive','negative','positive','outside'],
                physics_fps=fps,names=['L_Hip_y','L_Knee_y','L_Ankle_y','R_Hip_y','R_Knee_y','R_Ankle_y'],foot_contact_forces=contact)
    result=gait_cycle_moments(record,mass_kg=5)
    assert len(result['waveforms']['Hip'])==2
    assert min(result['cycle_counts'].values())>=3
    assert abs(max(result['waveforms']['Ankle'][0]['values'])-1)<.03
    assert result['waveforms']['Hip'][0]['values'][25]<-.95


def test_zero_contact_buffer_cannot_certify_swing_or_gait_cycles():
    from protomotions.robot_configs.human_model.common.strength_audit import gait_cycle_moments
    result=gait_cycle_moments({'foot_contact_forces':torch.zeros(300,1,2,3)})
    assert result['status']=='unavailable_zero_contact_trace'
    assert not result['waveforms']
