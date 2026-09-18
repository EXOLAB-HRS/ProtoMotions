"""v2 plant contracts; incompatible v1 coordinates must not silently run."""
import json
import numpy as np
import pytest
import torch
from protomotions.robot_configs.human_model.registry import get_model
from protomotions.robot_configs.human_model.profile import load_profile
from protomotions.robot_configs.human_model.dynamics import HumanJointModel
from protomotions.robot_configs.human_model.human_model_v2.model_config import robot_config


def test_v2_is_distinct_and_explicit(monkeypatch):
    monkeypatch.delenv('PROTOMOTIONS_HUMAN_MODEL',raising=False)
    m=get_model('human_model_v2');c=robot_config();p=load_profile('human_model_v2')
    assert m.runnable and m.asset_path('usd')!=get_model().asset_path('usd')
    assert c.number_of_actions==59 and len(c.kinematic_info.body_names)==26
    assert len([n for n in p['joints'] if any('_'+x+'_' in n for x in ['Hip','Knee','Ankle','Subtalar','Toe'])])==14
    with pytest.raises(ValueError,match='exact named'):HumanJointModel(p,list(load_profile()['joints']))
    with pytest.raises(ValueError,match='not been calibrated'):HumanJointModel(p,list(p['joints']),features=['strength'])
    with pytest.raises(ValueError,match='Unknown human model'):get_model('human_model_v20')


def test_v2_usd_frames_match_source_axes_and_limits():
    from pxr import Usd,UsdPhysics,Gf
    m=get_model('human_model_v2');stage=Usd.Stage.Open(str(m.asset_path('usd')));mapping=json.loads((m.root/'joint_map.json').read_text());p=load_profile('human_model_v2')
    for row in mapping['joints']:
        j=UsdPhysics.RevoluteJoint.Get(stage,'/smpl_humanoid/joints/'+row['name'])
        axis=Gf.Rotation(Gf.Quatd(j.GetLocalRot0Attr().Get())).TransformDir(Gf.Vec3d(1,0,0))
        np.testing.assert_allclose(axis,row['axis_local'],atol=1e-6)
        np.testing.assert_allclose(j.GetLocalPos0Attr().Get(),row['offset_parent'],atol=1e-7)
        np.testing.assert_allclose([j.GetLowerLimitAttr().Get(),j.GetUpperLimitAttr().Get()],p['joints'][row['name']]['rom_deg'],atol=1e-5)
        if 'source_axis' in row['source']:
            src=row['source'];expected=np.array(src['source_to_model_rotation'])@src['source_axis']*src['coordinate_sign'];expected/=np.linalg.norm(expected)
            np.testing.assert_allclose(axis,expected,atol=1e-6)
    assert len(mapping['joints'])==59


def test_v2_force_energy_and_directional_caps():
    p=load_profile('human_model_v2');names=list(p['joints']);m=HumanJointModel(p,names,dtype=torch.float64)
    torch.manual_seed(741);q=(m.lower+(m.upper-m.lower)*torch.rand(128,59,dtype=torch.float64)).requires_grad_();qd=torch.randn_like(q)
    total,active,passive=m.torques(torch.randn_like(q)*1e4,q,qd)
    negative,positive,_=m.strength_caps(q,qd)
    assert bool((active<=positive+1e-10).all() and (active>=-negative-1e-10).all())
    torch.testing.assert_close(total,active+passive)
    assert bool((m.damping_torque(qd)*qd<=1e-10).all())
    energy_gradient=torch.autograd.grad(m.potential_energy(q).sum(),q,retain_graph=True)[0]
    torch.testing.assert_close(m.elastic_torque(q),-energy_gradient,atol=1e-7,rtol=1e-7)
    # Jacobian symmetry checks conservative coupling, including angle transfer.
    jac=torch.autograd.functional.jacobian(lambda x:m.elastic_torque(x[None])[0],q[0].detach())
    torch.testing.assert_close(jac,jac.T,atol=1e-7,rtol=1e-7)


def test_trunk_candidate_serial_mapping_passivity_and_isolation():
    from pathlib import Path
    root=get_model('human_model_v2').root
    candidate=json.loads((root/'profiles/trunk_candidate.json').read_text())
    p=load_profile('human_model_v2');names=list(p['joints'])
    base=HumanJointModel(p,names,dtype=torch.float64)
    model=HumanJointModel(p,names,dtype=torch.float64,**candidate['human_model_parameters'])
    q=torch.zeros((1,len(names)),dtype=torch.float64,requires_grad=True)
    theta=np.deg2rad(9.)
    for axis,k in candidate['sources']['stiffness']['neutral_zone_stiffness_nm_per_degree'].items():
        indices=[names.index(f'{body}_{axis}') for body in ('Torso','Spine','Chest')]
        pose=q.detach().clone();pose[:,indices]=theta/3
        torque=model.elastic_torque(pose)-base.elastic_torque(pose)
        torch.testing.assert_close(torque[:,indices],torch.full((1,3),-k*9.,dtype=q.dtype))
    torch.manual_seed(1818)
    q=(torch.randn(32,len(names),dtype=torch.float64)*.2).requires_grad_();v=torch.randn_like(q)*20
    grad=torch.autograd.grad(model.potential_energy(q).sum(),q)[0]
    torch.testing.assert_close(model.elastic_torque(q),-grad)
    assert (model.damping_torque(v)*v<=0).all()
    total,active,passive=model.torques(q*1e6,q,v)
    assert (total.abs()<=model.backend_limit+1e-8).all()
    torch.testing.assert_close(active,base.torques(q*1e6,q,v)[1])
    other=[i for i,n in enumerate(names) if n not in candidate['human_model_parameters']['passive_joint_parameters']]
    torch.testing.assert_close(passive[:,other],base.passive_torque(q,v)[:,other])
    for invalid in (-1.,float('nan')):
        with pytest.raises(ValueError):
            HumanJointModel(p,names,passive_joint_parameters={'Torso_x':{'stiffness_nm_per_rad':[1.,1.],'damping_nms_per_rad':invalid}})


def test_trunk_candidate_config_serializes_parameters_and_preserves_legs():
    from protomotions.robot_configs.human_model.human_model_v2.model_config import configure_trunk_candidate
    from protomotions.robot_configs.human_model.common.integration import selected_model_parameters
    from protomotions.robot_configs.base import ControlType
    from protomotions.simulator.isaaclab.config import IsaacLabSimulatorConfig
    import pickle
    robot=robot_config();sim=IsaacLabSimulatorConfig(headless=True,num_envs=1,experiment_name='trunk_test');sim.sim=robot.simulation_params.isaaclab
    robot.control.control_type=ControlType.PROPORTIONAL
    before=pickle.dumps(robot.control.control_info['L_Hip_y'])
    configure_trunk_candidate(robot,sim)
    assert sim.sim.fps==480 and sim.sim.decimation==16
    assert robot.control.control_info['Torso_x'].stiffness==160.
    assert robot.control.control_info['Torso_x'].damping==2.
    assert before==pickle.dumps(robot.control.control_info['L_Hip_y'])
    restored=pickle.loads(pickle.dumps(robot))
    model=HumanJointModel(load_profile('human_model_v2'),robot.kinematic_info.dof_names,**selected_model_parameters(restored))
    assert float(model.k_positive[model.names.index('Torso_x')])>0


def test_v2_strength_work_sign_domain_and_backend_bound():
    p=load_profile('human_model_v2');m=HumanJointModel(p,list(p['joints']),dtype=torch.float64)
    q=torch.zeros(1,59,dtype=torch.float64);q[:,m.names.index('R_Knee_y')]=1.
    knee=m.names.index('R_Knee_y');velocity=torch.zeros_like(q)
    isometric=m.strength_caps(q,velocity)[0][0,knee]
    # Extension has negative torque: negative speed shortens; positive speed lengthens.
    velocity[:,knee]=-1.;con=m.strength_caps(q,velocity)[0][0,knee]
    velocity[:,knee]=1.;ecc=m.strength_caps(q,velocity)[0][0,knee]
    assert con<isometric and ecc>con
    velocity[:,knee]=-100
    n,p,flags=m.strength_caps(q,velocity,directional_domain=True)
    assert flags[0,knee,0] and flags[0,knee,1]
    velocity[:,knee]=-load_profile('human_model_v2')['active_strength_model']['directions']['knee_extension']['concentric_limit_rad_s']
    torch.testing.assert_close(n[0,knee],m.strength_caps(q,velocity)[0][0,knee])
    q=torch.linspace(-4.,4.,257,dtype=torch.float64)[:,None].expand(-1,59).clone()
    v=torch.linspace(-20.,20.,257,dtype=torch.float64)[:,None].expand_as(q)
    neg,pos,_=m.strength_caps(q,v)
    assert torch.isfinite(neg).all() and torch.isfinite(pos).all()
    assert (neg>=0).all() and (pos>=0).all()
    assert (torch.maximum(neg,pos)<=m.backend_limit).all()


def test_v2_strength_preserves_unmodelled_axes_and_correct_clinical_labels():
    p=load_profile('human_model_v2');m=HumanJointModel(p,list(p['joints']),dtype=torch.float64)
    q=torch.zeros(2,59,dtype=torch.float64);q[1]=.2;v=q*4
    neg,pos,_=m.strength_caps(q,v)
    for i,name in enumerate(m.names):
        if not any(name.endswith(x) for x in ['Hip_y','Knee_y','Ankle_y']):
            torch.testing.assert_close(neg[:,i],m.negative[i].expand(2))
            torch.testing.assert_close(pos[:,i],m.positive[i].expand(2))
    assert 'Positive clinical direction: adduction.' in p['joints']['R_Hip_x']['note']
    assert 'Positive clinical direction: internal rotation;' in p['joints']['R_Hip_z']['note'] or 'Positive clinical direction: internal rotation.' in p['joints']['R_Hip_z']['note']


def test_v2_runtime_rejects_asset_profile_drift():
    from protomotions.robot_configs.human_model.human_model_v2.assets.builder import validate_assets
    profile=load_profile('human_model_v2')
    assert validate_assets(profile)['joints']==59
    profile['joints']['L_Ankle_y']['rom_deg'][1]+=1
    with pytest.raises(ValueError,match='ROM mismatch'):validate_assets(profile)


def test_neutral_mass_com_inertia_are_preserved():
    from pxr import Usd,UsdPhysics,Gf
    def aggregate(model_id):
        stage=Usd.Stage.Open(str(get_model(model_id).asset_path('usd')));mass=0.;first=np.zeros(3);inertia=np.zeros((3,3))
        for p in stage.Traverse():
            if not p.HasAPI(UsdPhysics.MassAPI) or p.GetName().startswith('_joint_frame_'):continue
            m=UsdPhysics.MassAPI(p);value=float(m.GetMassAttr().Get());c=np.array(m.GetCenterOfMassAttr().Get())+np.array(p.GetAttribute('xformOp:transform').Get().ExtractTranslation())
            rot=np.array(Gf.Matrix3d(Gf.Rotation(Gf.Quatd(m.GetPrincipalAxesAttr().Get())))).T
            I=rot@np.diag(np.array(m.GetDiagonalInertiaAttr().Get()))@rot.T
            mass+=value;first+=value*c;inertia+=I+value*(c@c*np.eye(3)-np.outer(c,c))
        return mass,first/mass,inertia
    a=aggregate('human_model_v1');b=aggregate('human_model_v2')
    np.testing.assert_allclose(a[0],b[0],atol=1e-6,rtol=0)
    np.testing.assert_allclose(a[1],b[1],atol=1e-7,rtol=0)
    np.testing.assert_allclose(a[2],b[2],atol=1e-6,rtol=0)
