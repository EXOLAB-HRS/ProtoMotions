import math
import numpy as np
import pytest
import torch

from protomotions.robot_configs.human_model.dynamics import HumanJointModel
from protomotions.robot_configs.human_model.profile import load_profile


def test_fatigue_compartments_match_independent_ode_and_preserve_mass():
    from scipy.integrate import solve_ivp
    from protomotions.robot_configs.human_model.dynamics import fatigue_compartment_step
    state=torch.tensor([1.,0.,0.],dtype=torch.float64); reference=state.numpy().copy()
    f,r,l=.00912,.00094,10.
    for target,duration in [(.6,1.),(1.,10.),(0.,5.)]:
        def ode(t,s):
            rest,active,fatigued=s
            command=l*(min(target-active,rest) if active<target else target-active)
            return [-command+r*fatigued,command-f*active,f*active-r*fatigued]
        reference=solve_ivp(ode,[0,duration],reference,rtol=1e-10,atol=1e-12).y[:,-1]
        for _ in range(round(duration/.01)):
            state=fatigue_compartment_step(state,target,fatigue_rate=f,recovery_rate=r,dt=.01)
        np.testing.assert_allclose(state.numpy(),reference,atol=3e-6,rtol=0)
        assert (state>=0).all() and state.sum()==pytest.approx(1.)
    # Large caller intervals are subdivided before positivity can be lost.
    batched=torch.tensor([[1.,0.,0.],[0.,0.,1.]],dtype=torch.float64)
    result=fatigue_compartment_step(batched,[1.,0.],fatigue_rate=f,recovery_rate=r,dt=20.)
    assert (result>=0).all()
    np.testing.assert_allclose(result.sum(-1),1.)
    assert result[1,2]==pytest.approx(math.exp(-r*20),abs=1e-9)
    assert torch.equal(batched,torch.tensor([[1.,0.,0.],[0.,0.,1.]],dtype=torch.float64))


def test_fatigue_compartments_reject_invalid_state_and_parameters():
    from protomotions.robot_configs.human_model.dynamics import fatigue_compartment_step
    for state,target in [([1.,0.,.1],.2),([1.,-.1,.1],.2),([1.,0.,0.],1.1),([1.,0.,0.],[.2])]:
        with pytest.raises(ValueError):
            fatigue_compartment_step(torch.tensor(state),target,fatigue_rate=.01,recovery_rate=.001,dt=.01)
    for params in [{'fatigue_rate':-.01},{'recovery_rate':float('nan')},{'dt':0},{'tracking_rate':0}]:
        options=dict(fatigue_rate=.01,recovery_rate=.001,dt=.01);options.update(params)
        with pytest.raises(ValueError):
            fatigue_compartment_step(torch.tensor([1.,0.,0.]),.5,**options)


def test_joint_fatigue_reduces_torque_capacity_recovers_and_resets_selected_envs():
    p=load_profile();names=list(p['joints']);j=names.index('R_Knee_y')
    m=HumanJointModel(p,names,dtype=torch.float64,features=('fatigue',),fatigue_regions={'R_Knee_y':'Knee'})
    fresh=HumanJointModel(p,names,dtype=torch.float64)
    q=torch.zeros((2,69),dtype=torch.float64);request=torch.full_like(q,1e6)
    for _ in range(100):m.torques(request,q,q,dt=.01)
    peak=m.torques(request,q,q,dt=.01)[1][:,j].clone()
    for _ in range(900):total,active,passive,_,_,negative,positive,_=m._torque_components(request,q,q,dt=.01)
    assert (active[:,j]<peak).all()
    assert (positive[:,j]<m.positive[j]).all()
    assert (active<=positive+1e-12).all() and (active>=-negative-1e-12).all()
    mask=torch.arange(69)!=j
    assert torch.equal(active[:,mask],fresh.torques(request,q,q)[1][:,mask])
    torch.testing.assert_close(total,active+passive)
    fatigued=m._fatigue[...,2].clone()
    for _ in range(100):m.torques(q,q,q,dt=.1)
    assert (m._fatigue[:,j,1,2]<fatigued[:,j,1]).all()
    saved=m._fatigue.clone()
    first=m.torques(request,q,q,dt=.01)
    m._fatigue.copy_(saved)
    second=m.torques(request,q,q,dt=.01)
    assert all(torch.equal(a,b) for a,b in zip(first,second))
    retained=m._fatigue[1].clone();m.reset(torch.tensor([0]))
    assert torch.equal(m._fatigue[1],retained)
    assert (m._fatigue[0,...,0]==1).all() and not m._fatigue[0,...,1:].any()
    coactivation=q.clone();coactivation[:,j]=1
    m.torques(q,q,q,dt=.01,coactivation=coactivation)
    assert (m._fatigue[:,j,:,1]>0).all()
    with pytest.raises(ValueError,match='mapped'):
        m.torques(q,q,q,dt=.01,coactivation=q+1)
    for kwargs in [dict(features=('fatigue',)),dict(features=(),fatigue_regions={'R_Knee_y':'Knee'}),
                   dict(features=('fatigue','activation'),fatigue_regions={'R_Knee_y':'Knee'}),
                   dict(features=('fatigue',),fatigue_regions={'R_Knee_y':'unknown'})]:
        with pytest.raises(ValueError):HumanJointModel(p,names,**kwargs)


def model(features=()):
    p = load_profile()
    return HumanJointModel(p, list(p["joints"]), dtype=torch.float64, features=features)


def test_opt_in_keeps_default_outputs_and_rejects_unknown_feature():
    m = model(); q = torch.zeros(3, 69, dtype=torch.float64)
    total, active, passive = m.torques(q+100, q, q)
    torch.testing.assert_close(active, m.limit_active(q+100))
    torch.testing.assert_close(total, active + passive)
    with pytest.raises(ValueError):
        model(("typo",))


def test_coactivation_preserves_net_torque_with_asymmetric_directional_caps():
    baseline = model(('activation',)); candidate = model(('activation',))
    q = torch.zeros((3,69), dtype=torch.float64)
    requested = q.clone()
    j = candidate.names.index('L_Wrist_x')
    assert candidate.negative[j] != candidate.positive[j]
    requested[:,j] = torch.tensor([-2.,0.,3.])
    coactivation = torch.zeros_like(q); coactivation[:,j] = 1
    for _ in range(200):
        old = baseline.torques(requested,q,q,dt=.002)
        new = candidate.torques(requested,q,q,dt=.002,coactivation=coactivation)
    torch.testing.assert_close(new[1],old[1],atol=1e-12,rtol=1e-12)
    assert (candidate._activation[:,j,:] > baseline._activation[:,j,:]).any()
    assert (candidate._activation[:,j,:].amin(-1) > 0).all()
    assert ((candidate._activation >= 0) & (candidate._activation <= 1)).all()
    # Turning the opposing recruitment off retains finite relaxation dynamics.
    previous = candidate._activation.clone()
    candidate.torques(requested,q,q,dt=.002,coactivation=torch.zeros_like(q))
    assert (candidate._activation[:,j,:] <= previous[:,j,:]).all()
    assert (candidate._activation[:,j,:].amin(-1) > 0).all()
    retained = candidate._activation[2].clone()
    candidate.reset(torch.tensor([0,1]))
    assert not candidate._activation[:2].any()
    assert torch.equal(candidate._activation[2],retained)


def test_coactivation_zero_equivalence_capacity_and_invalid_inputs():
    plain = model(('strength','activation')); zero = model(('strength','activation'))
    q = torch.zeros((2,3,69),dtype=torch.float64)
    requested = torch.full_like(q,1e6)
    for sign in (1.,-1.,0.):
        a = plain.torques(sign*requested,q,q,dt=.003)
        b = zero.torques(sign*requested,q,q,dt=.003,coactivation=q)
        assert all(torch.equal(x,y) for x,y in zip(a,b))
    full = model(('strength','activation'))
    negative,positive,_ = full.strength_caps(q,q)
    for sign in (1.,-1.,0.):
        total,active,passive = full.torques(sign*requested,q,q,dt=.003,coactivation=q+1)
        assert ((active <= positive) & (active >= -negative)).all()
        torch.testing.assert_close(total,active+passive)
        assert ((full._activation >= 0) & (full._activation <= 1)).all()
    state = full._activation.clone()
    for invalid in (0.5,q-1,q+1.01,torch.full_like(q,float('nan'))):
        with pytest.raises(ValueError,match='coactivation'):
            full.torques(requested,q,q,dt=.003,coactivation=invalid)
        assert torch.equal(full._activation,state)
    with pytest.raises(ValueError,match='requires'):
        model().torques(q,q,q,coactivation=q)


def test_directional_domain_distinguishes_agonist_and_antagonist_limits():
    m = model(('strength',))
    q = torch.zeros((2, 3, 69), dtype=torch.float64)
    qd = torch.zeros_like(q)
    for side in ('L', 'R'):
        i = m.names.index(side+'_Ankle_y')
        qd[0,:,i] = math.radians(90)
        qd[1,:,i] = -math.radians(90)
    old = m.strength_caps(q, qd)
    new = m.strength_caps(q, qd, directional_domain=True)
    assert torch.equal(old[0], new[0]) and torch.equal(old[1], new[1])
    assert torch.equal(old[2], new[2].any(-1))
    for side in ('L', 'R'):
        i = m.names.index(side+'_Ankle_y')
        assert new[2][0,:,i,0].all() and not new[2][0,:,i,1].any()
        assert new[2][1,:,i,1].all() and not new[2][1,:,i,0].any()
    coupled = model(('strength', 'strength_coupling'))
    qd *= 0
    qd[..., m.names.index('L_Knee_y')] = .1
    flags = coupled.strength_caps(q, qd, directional_domain=True)[2]
    i = m.names.index('L_Ankle_y')
    assert flags[...,i,1].all() and not flags[...,i,0].any()
    assert torch.equal(flags.any(-1), coupled.strength_caps(q, qd)[2])


def test_source_cohort_selection_preserves_default_mixture_and_force_ceiling():
    profile=load_profile(); names=list(profile['joints'])
    models={sex:HumanJointModel(profile,names,dtype=torch.float64,
                               features=('strength','strength_coupling'),strength_cohort=sex)
            for sex in ('male','female','equal_sex')}
    q=torch.zeros((7,69),dtype=torch.float64)
    q[:,names.index('L_Knee_y')]=torch.linspace(0,2,7)
    qd=q*0
    caps={sex:m.strength_caps(q,qd) for sex,m in models.items()}
    for direction in (0,1):
        torch.testing.assert_close(caps['equal_sex'][direction],
                                   .5*(caps['male'][direction]+caps['female'][direction]))
    for m in models.values():
        total,_,_=m.torques(torch.full_like(q,1e6),q,qd)
        assert (total.abs()<=m.backend_limit+1e-8).all()
    with pytest.raises(ValueError,match='requires strength'):
        HumanJointModel(profile,names,strength_cohort='male')
    with pytest.raises(ValueError,match='strength_cohort'):
        HumanJointModel(profile,names,features=('strength',),strength_cohort='unknown')


def test_strength_reference_size_scales_selected_source_not_passive_or_other_joints():
    profile=load_profile(); names=list(profile['joints'])
    source=HumanJointModel(profile,names,dtype=torch.float64,features=('strength',),strength_cohort='male')
    resized=HumanJointModel(profile,names,dtype=torch.float64,features=('strength',),strength_cohort='male',strength_reference_size=(79,1.8))
    q=torch.full((2,69),.1,dtype=torch.float64);qd=q*0
    sn,sp,_=source.strength_caps(q,qd);rn,rp,_=resized.strength_caps(q,qd)
    source_size=source.candidate_parameters['strength']['cohorts'][0]
    ratio=79*1.8/(source_size['mass_kg']*source_size['height_m'])
    i=names.index('L_Ankle_y')
    torch.testing.assert_close(rn[:,i],sn[:,i]*ratio)
    torch.testing.assert_close(rp[:,i],sp[:,i]*ratio)
    j=names.index('L_Wrist_x')
    torch.testing.assert_close(rp[:,j],sp[:,j])
    torch.testing.assert_close(source.passive_torque(q,qd),resized.passive_torque(q,qd))
    total,_,_=resized.torques(q+1e6,q,qd)
    assert (total.abs()<=resized.backend_limit+1e-8).all()
    for size in ((0,1.8),(79,float('nan')),(-1,1.8),(79,)):
        with pytest.raises(ValueError):
            HumanJointModel(profile,names,features=('strength',),strength_reference_size=size)


@pytest.mark.parametrize('features',[(),('strength','strength_coupling','activation')])
def test_directional_strength_calibration_preserves_other_direction_and_passive(features):
    p=load_profile();names=list(p['joints']);kwargs=dict(dtype=torch.float64,features=features)
    baseline=HumanJointModel(p,names,**kwargs)
    candidate=HumanJointModel(p,names,active_strength_scale={'L_Ankle_y':[1,1.75],'R_Ankle_y':[0,1]},**kwargs)
    q=torch.full((4,69),.1,dtype=torch.float64);qd=q*0
    bn,bp,_=baseline.strength_caps(q,qd);cn,cp,_=candidate.strength_caps(q,qd)
    left=names.index('L_Ankle_y');right=names.index('R_Ankle_y')
    torch.testing.assert_close(cp[:,left],1.75*bp[:,left])
    torch.testing.assert_close(cn[:,left],bn[:,left])
    assert not cn[:,right].any()
    torch.testing.assert_close(cp[:,right],bp[:,right])
    torch.testing.assert_close(candidate.passive_torque(q,qd),baseline.passive_torque(q,qd))
    for sign in (-1,1):
        total,active,_=candidate.torques(q*0+sign*1e6,q,qd,dt=.1)
        assert (total.abs()<=candidate.backend_limit+1e-8).all()
        assert (active<=cp+1e-8).all() and (active>=-cn-1e-8).all()
    for scales in ({'missing':[1,1]},{'L_Ankle_y':[-1,1]},{'L_Ankle_y':[1,float('nan')]}):
        with pytest.raises(ValueError):HumanJointModel(p,names,active_strength_scale=scales,**kwargs)


def test_calibrated_robot_config_reaches_actual_mujoco_motor_force(monkeypatch):
    import mujoco
    from protomotions.robot_configs.smpl import SmplRobotConfig
    from protomotions.robot_configs.base import ControlType
    from protomotions.simulator.mujoco.config import MujocoSimulatorConfig
    from protomotions.simulator.mujoco.simulator import MujocoSimulator
    from protomotions.components.scene_lib import SceneLib,SceneLibConfig
    monkeypatch.delenv('PROTOMOTIONS_HUMAN_MODEL_FEATURES',raising=False)
    robot=SmplRobotConfig(human_model_parameters={'features':['strength'],
        'strength_cohort':'male','strength_reference_size':[79,1.8],
        'active_strength_scale':{'L_Ankle_y':[1,1.75]}})
    robot.control.control_type=ControlType.TORQUE
    sim=MujocoSimulator(MujocoSimulatorConfig(headless=True,num_envs=1,experiment_name='calibrated_plant_test'),
                        robot,None,torch.device('cpu'),SceneLib(SceneLibConfig()))
    sim._initialize_with_markers(None)
    force_model=sim._human_joint_model;i=force_model.names.index('L_Ankle_y')
    sim._common_actions.zero_();sim._common_actions[0,i]=1e6
    force_model.apply(sim)
    mujoco.mj_forward(sim.model,sim.data)
    expected=sim.human_applied_torques[0,sim.data_conversion.dof_convert_to_sim].numpy()
    np.testing.assert_allclose(sim.data.qfrc_actuator[6:75],expected,atol=2e-4,rtol=0)
    assert sim.human_active_torques[0,i]==sim.human_positive_caps[0,i]
    assert force_model.active_strength_scale[i,1]==1.75
    assert np.max(np.abs(expected))>0


@pytest.mark.parametrize('features', [(), ('passive_fit',)])
def test_passive_impedance_matches_force_derivatives_and_reciprocity(features):
    m = model(features)
    generator = torch.Generator().manual_seed(2200)
    q = m.lower + (.1 + .8*torch.rand(6,69,generator=generator,dtype=torch.float64))*(m.upper-m.lower)
    q[3:] *= 3  # numerical guard branches, not physiological postures
    qd = 30*torch.randn(q.shape,generator=generator,dtype=q.dtype)
    tangent = m.passive_impedance(q, qd)
    assert tangent['differentiable'].all()
    h = 1e-6
    derivative = torch.empty(6,69,69,dtype=q.dtype)
    for j in range(69):
        step = torch.zeros_like(q); step[:,j] = h
        derivative[:,:,j] = -(m.elastic_torque(q+step)-m.elastic_torque(q-step))/(2*h)
    stiffness = tangent['stiffness_nm_per_rad']
    torch.testing.assert_close(stiffness, derivative, rtol=1e-6, atol=1e-5)
    torch.testing.assert_close(stiffness, stiffness.transpose(-1,-2))
    assert torch.linalg.eigvalsh(stiffness).min() >= -1e-9
    assert torch.count_nonzero(stiffness - torch.diag_embed(stiffness.diagonal(dim1=-2,dim2=-1))) > 0
    damping = -(m.damping_torque(qd+h)-m.damping_torque(qd-h))/(2*h)
    torch.testing.assert_close(tangent['damping_nms_per_rad'], damping, rtol=1e-6, atol=1e-8)


def test_passive_impedance_marks_corners_and_excludes_missing_tissue_terms():
    m = model()
    q = torch.full((1,69), .1, dtype=torch.float64); qd = q*0
    wrist = int(torch.nonzero(m.damping > 0)[0])
    qd[:,wrist] = m.damping_cap / m.damping[wrist]
    assert not m.passive_impedance(q,qd)['differentiable'].item()
    qd[:,wrist] *= 2
    tangent = m.passive_impedance(q,qd)
    assert tangent['differentiable'].item()
    assert tangent['damping_nms_per_rad'][0,wrist] == 0
    assert (tangent['damping_nms_per_rad'][...,m.damping == 0] == 0).all()


def test_passive_guards_separate_coupled_spring_and_velocity_limits():
    m=model();q=torch.zeros((2,69),dtype=torch.float64);qd=q.clone()
    component=m.passive_component_names.index('L_hip_flexor')
    i=m.names.index('L_Hip_y')
    boundary=(math.log(m.exp_cap)-m.offset[component])/m.a[component,i]
    q[:,i]=boundary+torch.tensor([-.01,.01],dtype=q.dtype)
    wrist=m.names.index('L_Wrist_x')
    qd[1,wrist]=2*m.damping_cap/m.damping[wrist]
    q[1,wrist]=2*m.spring_cap/m.k_positive[wrist]
    flags=m.passive_guard_flags(q,qd)
    assert flags['exponential'][:,component].tolist()==[False,True]
    assert flags['spring'][:,wrist].tolist()==[False,True]
    assert flags['damping'][:,wrist].tolist()==[False,True]
    # Turning off a candidate energy term must not count as force saturation.
    m.energy_scale[component]=0
    assert not m.passive_guard_flags(q,qd)['exponential'][:,component].any()


def test_wrist_frequency_slope_conversion_uses_radian_velocity_only_when_selected():
    baseline, candidate = model(), model(('wrist_damping',))
    phase = torch.linspace(0,2*math.pi,1025,dtype=torch.float64)[:-1]
    qd = torch.zeros((len(phase),69),dtype=phase.dtype)
    wrist = candidate.names.index('L_Wrist_x')
    amplitude = math.radians(10)
    frequencies = torch.arange(3,13,dtype=phase.dtype)
    quadrature = []
    for frequency in frequencies:
        qd[:,wrist] = amplitude*2*math.pi*frequency*torch.cos(phase)
        torque = candidate.damping_torque(qd)[:,wrist]
        quadrature.append(-2*(torque*torch.cos(phase)).mean()/amplitude)
    slope = np.polyfit(frequencies.numpy(),torch.stack(quadrature).numpy(),1)[0]
    assert slope == pytest.approx((1.89+2.56)/2,rel=1e-12)
    assert baseline.damping[wrist] == pytest.approx(2.225)
    assert candidate.damping[wrist] == pytest.approx(2.225/(2*math.pi))
    other = [i for i,n in enumerate(candidate.names) if n not in ('L_Wrist_x','R_Wrist_x')]
    torch.testing.assert_close(candidate.damping[other],baseline.damping[other])
    from protomotions.robot_configs.human_model.validation import wrist_damping_assay
    for m, status in ((baseline,'fail'),(candidate,'pass')):
        result = wrist_damping_assay(m)
        for wrist_result in result['wrists'].values():
            assert wrist_result['definition_gate']['status'] == status
            assert wrist_result['biological_gate']['status'] == 'not_evaluated'


def test_strength_is_directional_uses_flexion_angle_and_reports_domain():
    m = model(("strength",)); q = torch.zeros(2, 69, dtype=torch.float64); qd = q.clone()
    i = m.names.index("L_Knee_y")
    q[:, i] = 1.2
    # Extension has negative generalized velocity but positive concentric velocity.
    qd[1, i] = -2
    neg, pos, _ = m.strength_caps(q, qd)
    assert neg[1, i] < neg[0, i]
    assert neg[0, i] != pos[0, i]
    # Independent scalar Eq.9 implementation, one joint/condition with both cohorts.
    reference = 0
    for c, mass, height in (([.163,1.258,1.133,1.517,3.952,.095],72.8,1.748),
                             ([.159,1.187,1.274,1.393,3.623,.173],62.1,1.606)):
        a = 2*c[3]*c[4]
        reference += .5*c[0]*mass*9.81*height*math.cos(c[1]*(1.2-c[2]))*(a+2*(c[4]-3*c[3]))/(a+2*(2*c[4]-4*c[3]))
    assert float(neg[1, i]) == pytest.approx(reference)
    q[:, i] = 8
    _, _, outside = m.strength_caps(q, qd)
    assert outside[:, i].all()
    total, _, _ = m.torques(q+1e6, q, qd)
    assert (total.abs() <= m.backend_limit+1e-8).all()


def test_activation_analytic_step_partial_reset_and_substep_invariance():
    fast, slow = model(("activation",)), model(("activation",))
    q = torch.zeros(2, 69, dtype=torch.float64)
    request = fast.positive.expand_as(q)
    for _ in range(100):
        _, a, _ = fast.torques(request, q, q, dt=.001)
    for _ in range(10):
        _, b, _ = slow.torques(request, q, q, dt=.01)
    torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(a, request*(1-math.exp(-.1/.01)))
    fast.reset(torch.tensor([0]))
    assert torch.count_nonzero(fast._activation[0]) == 0
    assert torch.count_nonzero(fast._activation[1]) > 0
    before = fast._activation.clone()
    _, a, _ = fast.torques(q, q, q, dt=.04)
    torch.testing.assert_close(fast._activation, before/math.e)
    with pytest.raises(ValueError):
        fast.torques(q, q, q)


def test_apply_reuses_components_without_double_stepping_activation():
    from types import SimpleNamespace as S
    from unittest.mock import patch
    from protomotions.robot_configs.base import ControlType
    features = ('strength', 'activation', 'passive_fit')
    applied, reference = model(features), model(features)
    q = torch.zeros(2,69,dtype=torch.float64)
    state = S(dof_pos=q, dof_vel=q.clone())
    state.convert_to_common = lambda _: state
    forces = []
    sim = S(_common_actions=q.clone(), _domain_randomization=None,
            _get_simulator_dof_state=lambda: state,
            robot_config=S(control=S(control_type=ControlType.TORQUE)),
            config=S(sim=S(fps=120)), data_conversion=S(dof_convert_to_sim=list(range(69))),
            _apply_simulator_torques=lambda value: forces.append(value.clone()))
    with patch.object(applied, 'strength_caps', wraps=applied.strength_caps) as caps:
        for command in (100., -100., 0.):
            sim._common_actions.fill_(command)
            expected = reference.torques(sim._common_actions, state.dof_pos, state.dof_vel, dt=1/120)
            applied.apply(sim)
            assert torch.equal(forces[-1], expected[0])
            assert torch.equal(applied._activation, reference._activation)
            assert torch.equal(sim.human_passive_torques, sim.human_elastic_torques+sim.human_damping_torques)
            assert torch.equal(forces[-1], sim.human_active_torques+sim.human_passive_torques)
        assert caps.call_count == 3


def test_strength_coupling_anchor_direction_domain_and_backend_ceiling():
    baseline=model(('strength',)); coupled=model(('strength','strength_coupling'))
    q=torch.zeros((3,69),dtype=torch.float64)
    knee=coupled.names.index('L_Knee_y');ankle=coupled.names.index('L_Ankle_y')
    q[:,knee]=torch.tensor([0,math.radians(50),math.radians(120)],dtype=q.dtype)
    bn,bp,_=baseline.strength_caps(q,q*0)
    cn,cp,domain=coupled.strength_caps(q,q*0)
    torch.testing.assert_close(bn,cn)
    torch.testing.assert_close(bp[1,ankle],cp[1,ankle])
    assert cp[0,ankle]>cp[1,ankle]>cp[2,ankle]
    assert not domain[:,ankle].any()
    qd=q*0;qd[:,ankle]=1
    assert coupled.strength_caps(q,qd)[2][:,ankle].all()
    q[:,knee]=-1
    assert coupled.strength_caps(q,q*0)[2][:,ankle].all()
    total,_,_=coupled.torques(q*0+1e6,q,qd)
    assert (total.abs()<=coupled.backend_limit+1e-8).all()
    with pytest.raises(ValueError,match='requires strength'):
        model(('strength_coupling',))


def test_fitted_passive_coupling_is_restoring_and_conservative():
    m = model(("passive_fit",))
    q = torch.zeros(2,69,dtype=torch.float64)
    knee, ankle = m.names.index('L_Knee_y'), m.names.index('L_Ankle_y')
    q[1,knee] = math.pi/3
    force=m.elastic_torque(q)
    assert force[0,ankle] > force[1,ankle]  # knee flexion relaxes GAS plantarflexion
    q.requires_grad_(True)
    gradient=torch.autograd.grad(m.potential_energy(q).sum(),q)[0]
    torch.testing.assert_close(m.elastic_torque(q),-gradient,atol=1e-10,rtol=1e-10)
    assert (m.energy_scale>=0).all()


@pytest.mark.parametrize('features', ['strength,activation', 'strength,activation,passive_fit', 'fatigue'])
def test_candidate_backend_selects_feature_and_exposes_substep_accounting(monkeypatch, features):
    from protomotions.robot_configs.smpl import SmplRobotConfig
    from protomotions.simulator.mujoco.config import MujocoSimulatorConfig
    from protomotions.simulator.mujoco.simulator import MujocoSimulator
    from protomotions.components.scene_lib import SceneLib, SceneLibConfig
    import mujoco
    monkeypatch.setenv("PROTOMOTIONS_HUMAN_MODEL_FEATURES", features)
    robot=SmplRobotConfig(human_model_parameters={'fatigue_regions':{'R_Knee_y':'Knee'}} if features=='fatigue' else {})
    if features=='fatigue':
        from protomotions.robot_configs.base import ControlType
        robot.control.control_type=ControlType.TORQUE
    sim = MujocoSimulator(MujocoSimulatorConfig(headless=True,num_envs=1,experiment_name="candidate_test"),
        robot, None, torch.device("cpu"), SceneLib(SceneLibConfig()))
    sim._initialize_with_markers(None)
    sim.model.opt.gravity[:] = 0
    sim.data.qpos[2] = 3
    sim.model.geom_contype[:] = 0; sim.model.geom_conaffinity[:] = 0
    sim._common_actions.zero_()
    if features=='fatigue':
        sim._common_actions[:,sim._human_joint_model.names.index('R_Knee_y')]=100
    mujoco.mj_forward(sim.model, sim.data)
    samples = []
    sim.human_model_trace_callback = lambda s, state: samples.append(s.human_applied_torques.clone())
    for _ in range(3):
        sim._physics_step()
    assert len(samples) == 3*sim.decimation
    assert np.isfinite(sim.data.qpos).all()
    assert sim._human_joint_model.features == set(features.split(','))
    if features=='fatigue':
        assert (sim._human_joint_model._fatigue[...,2]>0).any()
    torch.testing.assert_close(sim.human_applied_torques,
        sim.human_active_torques+sim.human_elastic_torques+sim.human_damping_torques)
    assert (sim.human_active_torques <= sim.human_positive_caps + 1e-5).all()
    assert (sim.human_active_torques >= -sim.human_negative_caps - 1e-5).all()


def test_legacy_explicit_pd_excludes_projectile_free_joints(monkeypatch):
    from protomotions.robot_configs.smpl import SmplRobotConfig
    from protomotions.simulator.mujoco.config import MujocoSimulatorConfig
    from protomotions.simulator.mujoco.simulator import MujocoSimulator
    from protomotions.components.scene_lib import SceneLib, SceneLibConfig
    monkeypatch.setenv('PROTOMOTIONS_HUMAN_MODEL','legacy')
    cfg=MujocoSimulatorConfig(headless=True,num_envs=1,experiment_name='legacy_pd_regression',use_implicit_pd=False)
    cfg.projectile.num_projectiles=2
    sim=MujocoSimulator(cfg,SmplRobotConfig(),None,torch.device('cpu'),SceneLib(SceneLibConfig()))
    sim._initialize_with_markers(None)
    assert sim.model.nq>7+69
    lower,upper=sim._get_simulator_dof_limits_for_verification()
    assert lower.shape==upper.shape==(69,)
    sim._cache_sim_order_pd_gains()
    sim._pd_targets_sim=sim.data.qpos[7:76].copy()
    sim.data.qvel[:]=0
    sim.data.qvel[75:]=1000
    sim._recompute_explicit_pd()
    np.testing.assert_allclose(sim._last_applied_torques,0,atol=1e-8)


@pytest.mark.parametrize('implicit',[False,True])
def test_legacy_joint_commands_are_not_multiplied_by_asset_gear(monkeypatch,implicit):
    import mujoco
    from protomotions.robot_configs.smpl import SmplRobotConfig
    from protomotions.simulator.mujoco.config import MujocoSimulatorConfig
    from protomotions.simulator.mujoco.simulator import MujocoSimulator
    from protomotions.components.scene_lib import SceneLib, SceneLibConfig
    from protomotions.robot_configs.base import ControlType
    monkeypatch.setenv('PROTOMOTIONS_HUMAN_MODEL','legacy')
    robot=SmplRobotConfig();robot.control.control_type=ControlType.BUILT_IN_PD
    cfg=MujocoSimulatorConfig(headless=True,num_envs=1,experiment_name='gear_units_regression',use_implicit_pd=implicit)
    sim=MujocoSimulator(cfg,robot,None,torch.device('cpu'),SceneLib(SceneLibConfig()))
    sim._initialize_with_markers(None)
    sim.data.qpos[7:76]=0;sim.data.qvel[:]=0
    targets=torch.zeros((1,69))
    # A 1 Nm proportional demand must produce 1 Nm at the joint in both modes.
    targets[0,0]=1/robot.control.control_info[sim._dof_names[0]].stiffness
    sim._apply_simulator_pd_targets(targets)
    mujoco.mj_forward(sim.model,sim.data)
    assert sim.data.qfrc_actuator[6]==pytest.approx(1.,rel=1e-6)
    measured=sim._get_simulator_dof_forces().dof_forces
    assert measured.shape==(1,69)
    assert measured[0,0].item()==pytest.approx(1.,rel=1e-6)
    np.testing.assert_allclose(sim.data.qfrc_actuator[7:75],0,atol=1e-8)


def test_mujoco_world_com_velocity_jacobian_reset_and_impulse(monkeypatch):
    import mujoco
    from protomotions.robot_configs.smpl import SmplRobotConfig
    from protomotions.simulator.mujoco.config import MujocoSimulatorConfig
    from protomotions.simulator.mujoco.simulator import MujocoSimulator
    from protomotions.simulator.base_simulator.simulator_state import ResetState
    from protomotions.components.scene_lib import SceneLib, SceneLibConfig
    monkeypatch.setenv('PROTOMOTIONS_HUMAN_MODEL','legacy')
    sim=MujocoSimulator(MujocoSimulatorConfig(headless=True,num_envs=1,experiment_name='velocity_frame_regression'),
                        SmplRobotConfig(),None,torch.device('cpu'),SceneLib(SceneLibConfig()))
    sim._initialize_with_markers(None)
    sim.data.qpos[3:7]=[2**-.5,0,0,2**-.5]
    sim.data.qvel[:75]=np.random.default_rng(1203).normal(0,.3,75)
    mujoco.mj_forward(sim.model,sim.data)
    bodies=sim._get_simulator_bodies_state()
    for body in range(1,1+sim._num_robot_bodies):
        jp=np.zeros((3,sim.model.nv));jr=np.zeros_like(jp)
        mujoco.mj_jacBodyCom(sim.model,sim.data,jp,jr,body)
        np.testing.assert_allclose(bodies.rigid_body_vel[0,body-1],jp@sim.data.qvel,atol=1e-6)
        np.testing.assert_allclose(bodies.rigid_body_ang_vel[0,body-1],jr@sim.data.qvel,atol=1e-6)
    root=sim._get_simulator_root_state()
    torch.testing.assert_close(root.root_vel,bodies.rigid_body_vel[:,0])
    torch.testing.assert_close(root.root_ang_vel,bodies.rigid_body_ang_vel[:,0])
    dof=sim._get_simulator_dof_state()
    desired_v=torch.tensor([[.2,-.3,.4]])
    desired_w=torch.tensor([[.7,.1,-.2]])
    reset=ResetState(root_pos=root.root_pos,root_rot=root.root_rot,
                     root_vel=desired_v,root_ang_vel=desired_w,dof_pos=dof.dof_pos,dof_vel=dof.dof_vel,
                     state_conversion=root.state_conversion)
    sim._set_simulator_env_state(reset)
    actual=sim._get_simulator_root_state()
    torch.testing.assert_close(actual.root_vel,desired_v)
    torch.testing.assert_close(actual.root_ang_vel,desired_w)
    dv=torch.tensor([[.1,.2,-.1]]);dw=torch.tensor([[-.2,.1,.3]])
    sim._apply_root_velocity_impulse(dv,dw,torch.tensor([0]))
    actual=sim._get_simulator_root_state()
    torch.testing.assert_close(actual.root_vel,desired_v+dv)
    torch.testing.assert_close(actual.root_ang_vel,desired_w+dw)
    sim._common_actions.copy_(sim._get_simulator_dof_state().dof_pos)
    sim._physics_step()
    root=sim._get_simulator_root_state();bodies=sim._get_simulator_bodies_state()
    # Derived body state must describe the integrated step, not its preceding
    # substep's kinematics cache.
    torch.testing.assert_close(root.root_pos,bodies.rigid_body_pos[:,0])
    torch.testing.assert_close(root.root_rot,bodies.rigid_body_rot[:,0])


def test_rest_augmented_fatigue_is_target_specific_and_conservative():
    from protomotions.robot_configs.human_model.dynamics import fatigue_compartment_step
    state=torch.tensor([[.5,0,.5],[.5,0,.5]],dtype=torch.float64)
    target=torch.tensor([0.,.5],dtype=torch.float64)
    original=fatigue_compartment_step(state,target,fatigue_rate=.015,recovery_rate=.00149,dt=.05)
    explicit=fatigue_compartment_step(state,target,fatigue_rate=.015,recovery_rate=.00149,dt=.05,rest_multiplier=1)
    augmented=fatigue_compartment_step(state,target,fatigue_rate=.015,recovery_rate=.00149,dt=.05,rest_multiplier=15)
    assert torch.equal(original,explicit)
    assert torch.equal(original[1],augmented[1])
    assert augmented[0,2] < original[0,2]
    torch.testing.assert_close(augmented.sum(-1),torch.ones(2,dtype=state.dtype))
    assert (augmented>=0).all()
    p=load_profile();names=list(p['joints']);j=names.index('R_Knee_y');q=torch.zeros(1,69,dtype=state.dtype)
    m=HumanJointModel(p,names,dtype=state.dtype,features=('fatigue',),fatigue_regions={'R_Knee_y':'Knee'},fatigue_rest_multiplier=15)
    m.torques(q,q,q,dt=.05);m._fatigue[0,j,1]=state[0]
    m.torques(q,q,q,dt=.05)
    torch.testing.assert_close(m._fatigue[0,j,1],augmented[0])
    with pytest.raises(ValueError):
        HumanJointModel(p,names,fatigue_rest_multiplier=15)
    for bad in [0,.5,float('nan')]:
        with pytest.raises(ValueError):
            fatigue_compartment_step(state,target,fatigue_rate=.015,recovery_rate=.00149,dt=.05,rest_multiplier=bad)
