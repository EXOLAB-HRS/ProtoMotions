"""Owned v2 pre-teacher batch ROM assay; no policy or retargeter."""
import json,math,time
from pathlib import Path
import numpy as np
import torch


def biological_audit(reference_path, output):
    """Frozen-plant literature audit; no parameter fitting or native rerun.

    Owned by e03_human_model_v2_biological_audit. Numeric band agreement is
    deliberately distinct from biological validation and protocol coverage.
    """
    import hashlib
    import csv
    from ..common.paths import WORKSPACE_ROOT, PACKAGE_ROOT
    from ..common.profile import load_profile
    from ..common.dynamics import HumanJointModel
    from .assets.builder import validate_assets
    refs = json.loads(Path(reference_path).read_text())
    for source in refs['sources'].values():
        actual = hashlib.sha256((WORKSPACE_ROOT/source['path']).read_bytes()).hexdigest()
        if actual != source['sha256']:
            raise ValueError('Reference source hash mismatch: '+source['path'])
    profile = load_profile('human_model_v2')
    asset_check = validate_assets(profile)
    model = HumanJointModel(profile, list(profile['joints']), dtype=torch.float64)
    names = model.names
    rows = []
    def add(identifier, family, value, unit, status, **extra):
        rows.append(dict(id=identifier, family=family, value=float(value), unit=unit,
                         status=status, **extra))
    mapping=json.loads((PACKAGE_ROOT/'human_model_v2/joint_map.json').read_text())
    mapped={j['name']:j for j in mapping['joints']}
    # +X rotates the downward right femur toward medial +Y. The source's
    # hip_adduction_r also has sign+1. Inherited v1 notes say abduction instead.
    for joint,expected,wrong in [('R_Hip_x','adduction','Positive clinical direction: abduction'),
                                 ('R_Hip_z','internal rotation','Positive clinical direction: external rotation')]:
        conflict=wrong in profile['joints'][joint].get('note','') and mapped[joint]['source']['coordinate_sign']==1
        add(joint+'_clinical_label','coordinate_semantics',int(conflict),'conflict',
            'candidate_failed_gate' if conflict else 'consistent',
            source='MyoLeg authored axis and coordinate sign',
            scope='Positive physical direction is '+expected+'; inherited label/strength-direction attribution requires audit before tuning')
    # An active endpoint cannot establish a passive anatomical limit. Only flag
    # anatomical ceilings too small even for the lower 2SD active endpoint.
    for ref in refs['rom_active_envelope']:
        limit = profile['joints'][ref['joint']]['rom_deg'][int(ref['sign'] > 0)]*ref['sign']
        low = ref['mean']-2*ref['sd']
        add(ref['id'], 'active_rom_envelope', limit, 'deg',
            'coverage_gap' if limit < low else 'necessary_envelope_only',
            reference_mean=ref['mean'], reference_sd=ref['sd'], lower_2sd=low,
            mean_shortfall_deg=max(0., ref['mean']-limit),
            z=(limit-ref['mean'])/ref['sd'], source='rom2019',
            scope='Scalar clinical-angle proxy; not a matched active-ROM trial or passive endpoint validation')
    w=refs['wang']; q=torch.zeros((1,len(names)),dtype=torch.float64)
    knee=names.index('R_Knee_y'); ankle=names.index('R_Ankle_y')
    q[:,knee]=math.radians(w['knee_deg']); qd=torch.zeros_like(q)
    requested=torch.zeros_like(q);requested[:,ankle]=1e4
    total,active,passive=model.torques(requested,q,qd)
    for component,value in [('active',active[0,ankle]),('total',total[0,ankle])]:
        z=(float(value)-w['mvc_mean_nm'])/w['mvc_sd_nm']
        add('ankle_mvc_'+component,'isometric_strength',value,'Nm',
            'within_2sd_proxy' if abs(z)<=2 else 'outside_2sd_proxy',z=z,
            reference_mean=w['mvc_mean_nm'],reference_sd=w['mvc_sd_nm'],source='wang2021',
            scope='Knee20 ankle0, saturated runtime torque; dynamometer axis/passive subtraction not exactly matched')
    lo,hi=w['ankle_sweep_deg'];count=501
    angles=torch.linspace(lo,hi,count,dtype=torch.float64)
    qs=q.expand(count,-1).clone();qs[:,ankle]=torch.deg2rad(angles)
    vs=torch.zeros_like(qs);vs[:,ankle]=-math.radians(w['speed_deg_s'])
    valid=(angles>=math.degrees(float(model.lower[ankle]))) & (angles<=math.degrees(float(model.upper[ankle])))
    passive_curve=model.passive_torque(qs,vs)[:,ankle]
    coverage=float(valid.double().mean())
    add('ankle_passive_protocol_coverage','passive_torque',coverage,'fraction',
        'protocol_not_reachable' if coverage<1 else 'covered',source='wang2021',
        model_min_deg=math.degrees(float(model.lower[ankle])),required_min_deg=lo,
        reachable_peak_nm=float(passive_curve[valid].max()),
        unphysical_extrapolation_peak_nm=float(passive_curve.max()),
        reference_mean=w['passive_peak_mean_nm'],reference_sd=w['passive_peak_sd_nm'],
        scope='Out-of-ROM algebraic values are diagnostic only and cannot receive a biological pass')
    # Read physical axes from authored USD, not just source metadata.
    from pxr import Usd,UsdPhysics,Gf
    stage=Usd.Stage.Open(str(PACKAGE_ROOT/'human_model_v2/assets/human_model_v2.usda'))
    joints={p.GetName():UsdPhysics.RevoluteJoint(p) for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)}
    for side in ('L','R'):
        joint=joints[side+'_Subtalar_x']
        axis=np.array(Gf.Rotation(Gf.Quatd(joint.GetLocalRot0Attr().Get())).TransformDir(Gf.Vec3d(1,0,0)))
        if axis[0]<0:axis=-axis
        medial=axis[1]*(-1 if side=='L' else 1)
        inc=math.degrees(math.atan2(axis[2],math.hypot(axis[0],axis[1])))
        dev=math.degrees(math.atan2(medial,axis[0]))
        for key,value in [('inclination',inc),('deviation',dev)]:
            ref=refs['subtalar'][side];z=(value-ref[key+'_mean'])/ref[key+'_sd']
            add(side+'_subtalar_'+key,'axis_geometry',value,'deg','frame_unmatched_proxy',
                z=z,reference_mean=ref[key+'_mean'],reference_sd=ref[key+'_sd'],
                within_2sd=abs(z)<=2,source='fernandez2020',scope=refs['subtalar']['frame'])
    toe=refs['toe_rom'];value=-profile['joints']['R_Toe_y']['rom_deg'][0]
    z=(value-toe['mean_deg'])/toe['sd_deg_approx']
    add('toe_dorsiflexion','rom_proxy',value,'deg','within_2sd_proxy' if abs(z)<=2 else 'outside_2sd_proxy',
        z=z,reference_mean=toe['mean_deg'],reference_sd=toe['sd_deg_approx'],source='toe2011',scope=toe['scope'])
    # Compare normalized capability shape with published fitted curves, not raw
    # participant SD. The 20% relative shape threshold is a project criterion.
    ref=json.loads((WORKSPACE_ROOT/refs['anderson']).read_text())['strength']
    scale=np.array([c['mass_kg']*ref['gravity_m_s2']*c['height_m'] for c in ref['cohorts']])
    for name,row in ref['directions'].items():
        i=names.index('R_'+row['joint']);c=np.array(row['coefficients']);sign=row['clinical_angle_sign']
        limits=np.sort(np.array(profile['joints']['R_'+row['joint']]['rom_deg'])*math.pi/180*sign)
        lower=max(limits[0],row['angle_domain_rad'][0]);upper=min(limits[1],row['angle_domain_rad'][1])
        a=np.linspace(lower,upper,9);speeds=np.array([0.,math.pi/3])
        assert speeds[-1]<=row['concentric_limit_rad_s']
        qa=q.expand(18,-1).clone();qa[:,i]=torch.tensor(np.repeat(a,2)/sign)
        va=torch.zeros_like(qa);va[:,i]=torch.tensor(np.tile(speeds,9)*row['sign'])
        neg,pos,_=model.strength_caps(qa,va);cap=(pos if row['sign']>0 else neg)[:,i].numpy().reshape(9,2)
        iso=np.maximum(np.cos(c[:,1]*(a[:,None]-c[:,2])),0)*c[:,0]*scale
        speed=speeds[None,:,None]
        factor=(2*c[:,3]*c[:,4]+speed*(c[:,4]-3*c[:,3]))/(2*c[:,3]*c[:,4]+speed*(2*c[:,4]-4*c[:,3]))
        predicted=(iso[:,None,:]*factor).mean(-1)
        ratio=float(np.mean(predicted[:,1]/predicted[:,0]));actual=float(np.mean(cap[:,1]/cap[:,0]))
        error=abs(actual/ratio-1)
        add(name+'_velocity_ratio','active_velocity_shape',actual,'ratio',
            'candidate_failed_gate' if error>.2 else 'reference_shape_agreement',
            reference_ratio=ratio,relative_error=error,threshold=.2,source=ref['source'],
            scope='60deg/s concentric divided by isometric; published fitted model, not population SD')
        refshape=predicted[:,0]/predicted[:,0].max();actualshape=cap[:,0]/cap[:,0].max()
        rmse=float(np.sqrt(np.mean((refshape-actualshape)**2)))
        add(name+'_angle_shape','active_angle_shape',rmse,'normalized_rmse',
            'candidate_failed_gate' if rmse>.2 else 'reference_shape_agreement',
            threshold=.2,reference_normalized=refshape.tolist(),model_normalized=actualshape.tolist(),
            angles_clinical_deg=np.rad2deg(a).tolist(),source=ref['source'],
            scope='Shape normalized by own peak over shared ROM; project tolerance, not clinical certification')
    qs=q.expand(2,-1).clone();qs[:,ankle]=math.radians(-10);qs[:,knee]=torch.tensor([0.,math.pi/2])
    change=float((model.elastic_torque(qs)[1,ankle]-model.elastic_torque(qs)[0,ankle]).abs())
    add('ankle_passive_knee_dependence','biarticular_coupling',change,'Nm',
        'missing_dependency' if change<1e-8 else 'dependency_present_unvalidated',
        source=profile['sources']['SILDER2007']['url'],scope='Structural test at ankle -10, knee0/90. No population effect-size acceptance without matched data.')
    failed=any(r['status'] in {'candidate_failed_gate','coverage_gap','protocol_not_reachable','missing_dependency','outside_2sd_proxy'} for r in rows)
    result=dict(model_id='human_model_v2',status='candidate_failed_gate' if failed else 'partial_evidence_not_certified',
        scope='Literature-based frozen-plant screening audit; analytical runtime calls and USD inspection, no new IsaacLab experiment, no parameter fitting',
        criterion=refs['criterion'],asset_check=asset_check,rows=rows,
        unverified=['Subject-matched anatomical landmark frames and knee moving-axis/coupled-translation accuracy',
                    'Hip/ankle/toe independent passive torque curves across matched full ROM',
                    'Subtalar active strength, damping, activation delay; upper-body independent biological validation',
                    'Whole-body locomotion, self-collision, teacher learning'],
        calibration_evidence='e02 population_scale.json knee/toe curves reuse calibration cohorts; not independent passes',
        source_records=refs['sources'])
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    with (output/'metrics_summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['id','family','value','unit','status','z','reference_mean','reference_sd','scope'],extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)
    return result


def native_suite(sim,app,output,seed,protocol,initial):
    """One independently clamped test axis/direction per environment.

    Other-axis clamps are explicit external test supports. The tested axis keeps
    the original limits and receives human torque only, except labelled zero-
    strength limit probes. Measures every physical step, saves all-axis 30Hz trace.
    """
    if protocol.get('assay')=='smpl2hm_tracking':return native_retarget_tracking(sim,app,output,seed,protocol,initial)
    if protocol.get('assay')=='strength':return native_strength(sim,app,output,seed,protocol,initial)
    if protocol.get('assay')=='benchmark':return native_benchmark(sim,app,output,seed,protocol,initial)
    if protocol.get('assay')=='supported_feet':return native_supported_feet(sim,app,output,seed,protocol,initial)
    # reset_envs converts ResetState in place; restore its convention before edits.
    initial.convert_to_common(sim.data_conversion)
    model=sim._human_joint_model;device=sim.device;view=sim._robot.root_physx_view
    names=model.names;n=len(names);N=sim.num_envs;fps=sim.config.sim.fps;dt=1/fps
    assert N==2*n
    order=sim.data_conversion.dof_convert_to_common
    env=torch.arange(N,device=device);focus=env//2;direction=torch.where(env%2==0,-1.,1.)
    nominal=torch.stack((model.lower,model.upper),-1).unsqueeze(0).expand(N,-1,-1).clone()
    pose=initial.dof_pos.clone();pose.zero_();initial.root_pos[:,2]=1.8
    initial.root_pos[:,0]=3.+3.*(env%math.ceil(math.sqrt(N)))
    initial.root_pos[:,1]=3.+3.*(env//math.ceil(math.sqrt(N)))
    pose[:,names.index('L_Shoulder_x')]=-math.pi/2;pose[:,names.index('R_Shoulder_x')]=math.pi/2
    for i in range(N):
        name=names[i//2]
        if name.endswith('Hip_y') and i%2==0:pose[i,names.index(name.replace('Hip_y','Knee_y'))]=math.pi/2
        if name.endswith('Knee_y'):pose[i,names.index(name.replace('Knee_y','Hip_y'))]=-math.pi/2
    pose[env,focus]=0
    initial.dof_pos=pose.clone();initial.dof_vel.zero_()
    limits=nominal.clone();width=math.radians(.005)
    limits[:,:,0]=torch.maximum(model.lower,pose-width);limits[:,:,1]=torch.minimum(model.upper,pose+width);limits[env,focus]=nominal[env,focus]
    native=torch.empty_like(limits);native[:,order]=limits
    view.set_dof_limits(native.cpu(),torch.arange(N,dtype=torch.int32))
    observed=view.get_dof_limits().to(device)[:,order]
    assert torch.allclose(observed[env,focus],nominal[env,focus],atol=1e-6,rtol=0)
    sim.reset_envs(initial)
    if float((sim.get_dof_state().dof_pos-pose).abs().max())>1e-5:
        raise RuntimeError("Initial pose readback/order mismatch")
    lower=model.lower[focus];upper=model.upper[focus];bound=torch.where(direction<0,lower,upper);endpoint=bound-direction*math.radians(.1)
    locked=torch.maximum(model.negative[focus],model.positive[focus])==0
    inertia=view.get_generalized_mass_matrices()[:,order][:,:,order].diagonal(dim1=-2,dim2=-1)+view.get_dof_armatures().to(device)[:,order]
    I=inertia[env,focus].clamp_min(.001);kp=torch.full_like(I,2000.)
    for i in range(N):
        if names[i//2] in ('Torso_z','Spine_z','Chest_z'):kp[i]=500.
    kd=2*torch.sqrt(kp*I)
    ramp=6.;duration=19.;knots=[(0.,0.),(1.,0.),(7.,1.),(8.5,1.),(9.,2.),(10.,2.),(16.,0.),(19.,0.)]
    challenge=bound+direction*math.radians(3.)
    def target(t):
        if t>=19:return bound*0,bound*0,bound*0
        i=next(i for i in range(len(knots)-1) if knots[i][0]<=t<knots[i+1][0]);a,qa=knots[i];b,qb=knots[i+1]
        values=[bound*0,endpoint,challenge];x=(t-a)/(b-a);delta=values[int(qb)]-values[int(qa)]
        return values[int(qa)]+delta*(10*x**3-15*x**4+6*x**5),delta*(30*x*x-60*x**3+30*x**4)/(b-a),delta*(60*x-180*x*x+120*x**3)/(b-a)**2
    metrics={k:torch.zeros(N,device=device) for k in ['rom_excess','cap_error','sum_error','endpoint_error','recovery_error','hold_speed','hold_torque_min','hold_torque_max','hold_q_min','hold_q_max','boundary_distance','native_effort_error']}
    for k in ['hold_torque_min','hold_q_min','boundary_distance']:metrics[k].fill_(float('inf'))
    for k in ['hold_torque_max','hold_q_max']:metrics[k].fill_(-float('inf'))
    records={k:[] for k in ['time','q','qd','target','requested','active','passive','applied','external_probe','negative_cap','positive_cap','outside_domain']}
    start=time.perf_counter();maxroot=0.
    for step in range(round(duration*fps)):
        t=step*dt;desired,velocity,feedforward=target(t);state=sim.get_dof_state();q=state.dof_pos;qd=state.dof_vel
        gravity=view.get_gravity_compensation_forces()[:,order]
        passive=model.passive_torque(q,qd)
        command=torch.zeros_like(q)
        command[env,focus]=kp*(desired-q[env,focus])+kd*(velocity-qd[env,focus])+I*feedforward+gravity[env,focus]-passive[env,focus]
        sim._common_actions=command;sim._apply_control();sim._scene.write_data_to_sim()
        probe=torch.where(locked & (t>=8.5) & (t<10),.5*direction,torch.zeros_like(direction))
        native_effort=sim.human_applied_torques.clone();native_effort[env,focus]+=probe
        converted=torch.empty_like(native_effort);converted[:,order]=native_effort;view.set_dof_actuation_forces(converted,env.to(torch.int32))
        metrics['native_effort_error']=torch.maximum(metrics['native_effort_error'],(view.get_dof_actuation_forces()[:,order]-native_effort).abs().max(1).values)
        sim._sim.step(render=False);sim._scene.update(dt=dt)
        after=sim.get_dof_state();aq=after.dof_pos;av=after.dof_vel
        if not bool(torch.isfinite(aq).all() and torch.isfinite(av).all()):raise RuntimeError('Nonfinite batch state')
        metrics['rom_excess']=torch.maximum(metrics['rom_excess'],torch.maximum(model.lower-aq,aq-model.upper).clamp_min(0).max(1).values)
        metrics['cap_error']=torch.maximum(metrics['cap_error'],torch.maximum(sim.human_active_torques-sim.human_positive_caps,-sim.human_active_torques-sim.human_negative_caps).clamp_min(0).max(1).values)
        metrics['sum_error']=torch.maximum(metrics['sum_error'],(sim.human_applied_torques-sim.human_active_torques-sim.human_passive_torques).abs().max(1).values)
        if 7.5<=t<8.5:
            for key,value in [('endpoint_error',(aq[env,focus]-endpoint).abs()),('hold_speed',av[env,focus].abs())]:metrics[key]=torch.maximum(metrics[key],value)
            for prefix,value in [('hold_q',aq[env,focus]),('hold_torque',sim.human_active_torques[env,focus])]:
                metrics[prefix+'_min']=torch.minimum(metrics[prefix+'_min'],value);metrics[prefix+'_max']=torch.maximum(metrics[prefix+'_max'],value)
        if 9.5<=t<10:metrics['boundary_distance']=torch.minimum(metrics['boundary_distance'],(aq[env,focus]-bound).abs())
        if 18<=t:metrics['recovery_error']=torch.maximum(metrics['recovery_error'],aq[env,focus].abs())
        maxroot=max(maxroot,float((sim.get_root_state().root_pos-initial.root_pos).abs().max()))
        if step%(fps//30)==0:
            records['time'].append((step+1)*dt)
            for key,value in [('q',aq),('qd',av),('target',desired),('requested',command),('active',sim.human_active_torques),('passive',sim.human_passive_torques),('applied',sim.human_applied_torques),('external_probe',probe),('negative_cap',sim.human_negative_caps),('positive_cap',sim.human_positive_caps),('outside_domain',sim.human_strength_outside_domain)]:records[key].append(value.cpu().numpy().copy())
        if step%fps==0:print('V2_BATCH_SECOND',round(t),'ROM_DEG',float(torch.rad2deg(metrics['rom_excess']).max()),flush=True)
    elapsed=time.perf_counter()-start
    measured={k:v.cpu().numpy() for k,v in metrics.items()};rows=[]
    for i in range(N):
        deg=lambda k:float(np.rad2deg(measured[k][i]))
        lock=bool(locked[i]);peak=float(max(model.negative[i//2],model.positive[i//2]))
        checks=dict(rom=deg('rom_excess')<=.1,caps=float(measured['cap_error'][i])<=.0002,torque_sum=float(measured['sum_error'][i])<=.0002,
            endpoint=lock or deg('endpoint_error')<=2,recovery=lock or deg('recovery_error')<=2,
            hold_speed=lock or deg('hold_speed')<=5,
            hold_ripple=lock or float(np.rad2deg(measured['hold_q_max'][i]-measured['hold_q_min'][i]))<=.2,
            torque_ripple=lock or float(measured['hold_torque_max'][i]-measured['hold_torque_min'][i])<=max(.5,.05*peak),
            boundary=deg('boundary_distance')<=.01,native_effort=float(measured['native_effort_error'][i])<=.0002)
        rows.append(dict(joint=names[i//2],direction=-1 if i%2==0 else 1,kind='external_lock_probe' if lock else 'active',status='passed' if all(checks.values()) else 'candidate_failed_gate',checks=checks,metrics={k:float(v[i]) for k,v in measured.items()}))
    stem=Path(output)/f'human_model_v2_v2_suite_seed{seed}_raw'
    np.savez_compressed(stem.with_suffix('.npz'),**{k:np.asarray(v) for k,v in records.items()},names=names,limits=nominal.cpu().numpy(),fixture_limits=observed.cpu().numpy())
    result=dict(model_id='human_model_v2',scenario='v2_suite',seed=seed,protocol=protocol,physics_fps=fps,environment_count=N,elapsed_s=elapsed,env_steps_per_s=N*duration*fps/elapsed,root_error_m=maxroot,
        direction_results=rows,engine_gate=dict(status='passed' if all(r['status']=='passed' for r in rows) and maxroot<1e-5 else 'candidate_failed_gate'),shutdown='pending',
        scope='Independent joint tests under explicit other-axis clamps, gravity ON, feet airborne; zero-strength axes use external .5Nm probe; no teacher')
    stem.with_suffix('.json').write_text(json.dumps(result,indent=2));print('VIDEO_CAPTURE_SAVED',flush=True)
    sim._sim.clear_all_callbacks()
    from isaaclab.sim import SimulationContext
    SimulationContext.clear_instance();app.close(wait_for_replicator=False)


def population_audit(output):
    """Recheck descriptive population scale, explicitly separating calibration."""
    from ..common.paths import WORKSPACE_ROOT
    from ..common.profile import load_profile
    from ..common.dynamics import HumanJointModel
    from ..common.metrics import population_band_metrics
    src=WORKSPACE_ROOT/'output/260914/e01_human_model_population/sources'
    reports={}
    for model_id in ['human_model_v1','human_model_v2']:
        p=load_profile(model_id);m=HumanJointModel(p,list(p['joints']),dtype=torch.float64);names=m.names
        knee=json.loads((src/'kaneda2020_population.json').read_text());r=knee['passive'];q=torch.zeros((len(r['mean_nm']),len(names)),dtype=torch.float64)
        q[:,names.index('R_Hip_y')]=-math.radians(r['hip_flexion_deg']);q[:,names.index('R_Knee_y')]=torch.deg2rad(torch.tensor(r['clinical_flexion_deg'],dtype=torch.float64))
        pred=m.elastic_torque(q)[:,names.index('R_Knee_y')].numpy()
        morin=json.loads((src/'morin2023_population.json').read_text());strength=[]
        for row in morin['records']:
            if row['joint'] not in names or 'Hip' in row['joint']:continue
            j=names.index(row['joint']);value=float(m.negative[j] if row['direction']<0 else m.positive[j]);strength.append(dict(joint=row['joint'],direction=row['direction'],predicted_nm=value,mean_nm=row['mean_nm'],sd_nm=row['sd_nm'],z=(value-row['mean_nm'])/row['sd_nm']))
        toe=json.loads((src/'toe2025_population.json').read_text());indices=[i for i,n in enumerate(toe['n_per_angle']) if n>=3];tq=torch.zeros((len(indices),len(names)),dtype=torch.float64);tq[:,names.index('R_Toe_y')]=-torch.deg2rad(torch.tensor([toe['angle_deg'][i] for i in indices],dtype=torch.float64));tp=m.elastic_torque(tq)[:,names.index('R_Toe_y')].numpy()
        reports[model_id]=dict(knee=population_band_metrics(pred,r['mean_nm'],r['sd_nm']),knee_predictions_nm=pred.tolist(),
            knee_heldout_angles=population_band_metrics(pred[1::2],np.array(r['mean_nm'])[1::2],np.array(r['sd_nm'])[1::2]),
            toe=population_band_metrics(tp,[toe['mean_nm'][i] for i in indices],[toe['sd_nm'][i] for i in indices]),strength=strength)
    reports['scope']='Descriptive scale of generalized joint torques. Knee/toe reuse calibration cohorts; held-out angles are not held-out people. New oblique coordinates and subtalar strength have no independent population certification. Upper limb remains v1 proxy.'
    Path(output).write_text(json.dumps(reports,indent=2)+'\n');return reports


def native_benchmark(sim,app,output,seed,protocol,initial):
    """Matched warm-start torque plant throughput and partial-reset isolation."""
    initial.convert_to_common(sim.data_conversion)
    model=sim._human_joint_model;view=sim._robot.root_physx_view;device=sim.device;N=sim.num_envs
    env=torch.arange(N,device=device);initial.root_pos[:,0]=3+3*(env%math.ceil(math.sqrt(N)));initial.root_pos[:,1]=3+3*(env//math.ceil(math.sqrt(N)));initial.root_pos[:,2]=1.5
    initial.dof_pos.zero_();initial.dof_pos[:,model.names.index('L_Shoulder_x')]=-math.pi/2;initial.dof_pos[:,model.names.index('R_Shoulder_x')]=math.pi/2
    pose=initial.dof_pos.clone();sim.reset_envs(initial);order=sim.data_conversion.dof_convert_to_common;fps=sim.config.sim.fps
    warmup=240;steps=1200;peak=0.;cap=0.;start=0.
    for step in range(warmup+steps):
        if step==warmup:
            if device.type=='cuda':torch.cuda.synchronize()
            start=time.perf_counter()
        state=sim.get_dof_state();target=pose.clone()
        target[:,model.names.index('L_Hip_y')]-=.1*math.sin(step/fps)
        target[:,model.names.index('R_Hip_y')]+=.1*math.sin(step/fps)
        command=40*(target-state.dof_pos)-3*state.dof_vel+view.get_gravity_compensation_forces()[:,order]
        sim._common_actions=command;sim._apply_control();sim._scene.write_data_to_sim();sim._sim.step(render=False);sim._scene.update(dt=1/fps)
    if device.type=='cuda':torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    before=sim.get_dof_state().dof_pos.clone()
    from protomotions.simulator.base_simulator.simulator_state import ResetState,StateConversion
    partial=ResetState(root_pos=initial.root_pos[:1].clone(),root_rot=torch.tensor([[0.,0.,0.,1.]],device=device),root_vel=initial.root_vel[:1].clone(),root_ang_vel=initial.root_ang_vel[:1].clone(),dof_pos=pose[:1].clone(),dof_vel=pose[:1]*0,state_conversion=StateConversion.COMMON)
    sim.reset_envs(partial,env_ids=torch.tensor([0],device=device));after=sim.get_dof_state().dof_pos
    isolation=float((after[1:]-before[1:]).abs().max());reset_error=float((after[:1]-pose[:1]).abs().max())
    finite=bool(torch.isfinite(before).all());result=dict(model_id=model.profile_id,scenario='benchmark',seed=seed,protocol=protocol,physics_fps=fps,environment_count=N,measured_steps=steps,elapsed_s=elapsed,env_steps_per_s=N*steps/elapsed,
        partial_reset_other_env_error_rad=isolation,reset_error_rad=reset_error,engine_gate=dict(status='passed' if finite and isolation<1e-6 and reset_error<1e-6 else 'candidate_failed_gate'),shutdown='pending',scope='Matched GPU explicit PD torque loop; not policy learning throughput')
    stem=Path(output)/f'{protocol["model_id"]}_v2_suite_seed{seed}_raw';stem.with_suffix('.json').write_text(json.dumps(result,indent=2))
    print('BENCHMARK_COMPLETE',result,flush=True);sim._sim.clear_all_callbacks()
    from isaaclab.sim import SimulationContext
    SimulationContext.clear_instance();app.close(wait_for_replicator=False)


def native_supported_feet(sim,app,output,seed,protocol,initial):
    """External pelvis harness supports half weight; feet bear the remainder.

    Harness has horizontal position/orientation feedback, constant upward force bias plus vertical velocity damping,
    and no vertical position servo. This is a loaded foot assay, not balance.
    """
    initial.convert_to_common(sim.data_conversion);model=sim._human_joint_model;view=sim._robot.root_physx_view;device=sim.device
    assert sim.num_envs==1 and not sim._robot.is_fixed_base
    initial.root_pos[:]=torch.tensor([3.,3.,1.0],device=device);initial.root_rot[:]=torch.tensor([0.,0.,0.,1.],device=device);initial.dof_pos.zero_()
    for s,sign in [('L',-1),('R',1)]:
        for j,a in [('Hip_y',-5),('Knee_y',10),('Ankle_y',-5),('Shoulder_x',sign*80)]:initial.dof_pos[:,model.names.index(s+'_'+j)]=math.radians(a)
    pose=initial.dof_pos.clone();sim.reset_envs(initial);order=sim.data_conversion.dof_convert_to_common
    foot_axes=[i for i,n in enumerate(model.names) if any('_'+j+'_' in n for j in ('Ankle','Subtalar','Toe'))]
    limits=torch.stack((model.lower,model.upper),-1)[None].clone();nominal=limits.clone()
    limits[:,:,0]=torch.maximum(model.lower,pose-math.radians(.005));limits[:,:,1]=torch.minimum(model.upper,pose+math.radians(.005));limits[:,foot_axes]=nominal[:,foot_axes]
    native=torch.empty_like(limits);native[:,order]=limits;view.set_dof_limits(native.cpu(),torch.tensor([0],dtype=torch.int32))
    arm=view.get_dof_armatures().to(device)[:,order];weight=float(view.get_masses().sum())*9.81;pelvis=sim._robot.body_names.index('Pelvis');fps=sim.config.sim.fps;dt=1/fps
    records={k:[] for k in ['time','q','qd','target','root_pos','root_rot','contact_force','contact_body_force','harness_force','harness_torque','rom_excess','cap_error','sum_error']}
    peak_penetration=0.;peak_rom=0.;peak_cap=0.
    for step in range(12*fps):
        t=step/fps;state=sim.get_dof_state();root=sim.get_root_state();target=pose.clone()
        if 3<=t<=9:
            x=(t-3)/6;wave=math.sin(2*math.pi*x)*math.sin(math.pi*x)**2
            for s,sign in [('L',1),('R',-1)]:
                target[:,model.names.index(s+'_Subtalar_x')]+=sign*math.radians(2)*wave
                target[:,model.names.index(s+'_Ankle_y')]+=math.radians(3)*wave
                target[:,model.names.index(s+'_Toe_y')]-=math.radians(3)*wave
        M=view.get_generalized_mass_matrices()[:,6:,6:][:,order][:,:,order]+torch.diag_embed(arm)
        command=torch.zeros_like(target)
        command[:,foot_axes]=200*(target-state.dof_pos)[:,foot_axes]-5*state.dof_vel[:,foot_axes]
        command[:,foot_axes]+=view.get_gravity_compensation_forces()[:,order][:,foot_axes]-model.passive_torque(state.dof_pos,state.dof_vel)[:,foot_axes]
        force=torch.zeros((1,1,3),device=device);torque=force.clone();force[0,0,:2]=2000*(torch.tensor([3.,3.],device=device)-root.root_pos[0,:2])-150*root.root_vel[0,:2];force[0,0,2]=.5*weight-250*root.root_vel[0,2]
        quaternion=root.root_rot[0];torque[0,0]=-1000*2*quaternion[:3]*torch.sign(quaternion[3])-100*root.root_ang_vel[0]
        sim._robot.set_external_force_and_torque(force,torque,body_ids=[pelvis],is_global=True)
        sim._common_actions=command;sim._apply_control();sim._scene.write_data_to_sim();sim._sim.step(render=False);sim._scene.update(dt=dt)
        after=sim.get_dof_state();rom=torch.maximum(model.lower-after.dof_pos,after.dof_pos-model.upper).clamp_min(0).max();cap=torch.maximum(sim.human_active_torques-sim.human_positive_caps,-sim.human_active_torques-sim.human_negative_caps).clamp_min(0).max();peak_rom=max(peak_rom,float(rom));peak_cap=max(peak_cap,float(cap))
        if step%(fps//30)==0:
            for sensor in sim._contact_sensor_map.values():
                _,_,_,sep,counts,starts=sensor.contact_physx_view.get_contact_data(dt)
                for count,begin in zip(counts.flatten().tolist(),starts.flatten().tolist()):
                    if count:peak_penetration=min(peak_penetration,float(sep[begin:begin+count].min()))
            forces=sim.get_bodies_contact_buf().rigid_body_contact_forces
            records['time'].append((step+1)*dt)
            for key,value in [('q',after.dof_pos),('qd',after.dof_vel),('target',target),('root_pos',sim.get_root_state().root_pos),('root_rot',sim.get_root_state().root_rot),('contact_force',forces.sum(1)),('contact_body_force',forces),('harness_force',force),('harness_torque',torque)]:records[key].append(value[0].cpu().numpy().copy())
            for k,v in [('rom_excess',rom),('cap_error',cap),('sum_error',(sim.human_applied_torques-sim.human_active_torques-sim.human_passive_torques).abs().max())]:records[k].append(float(v))
    d={k:np.array(v) for k,v in records.items()};settled=d['time']>=10;contact=d['contact_force'][settled,2].mean();footids=[i for i,n in enumerate(sim.robot_config.kinematic_info.body_names) if n.endswith(('Ankle','Toe'))];foot=d['contact_body_force'][settled][:,footids,2].sum(1).mean();expected=weight-d['harness_force'][settled,0,2].mean();error=abs(contact/expected-1)
    foot_ripple=float(np.rad2deg(np.ptp(d['q'][settled][:,foot_axes],axis=0)).max());foot_speed=float(np.rad2deg(abs(d['qd'][settled][:,foot_axes])).max())
    checks=dict(foot_hold_ripple=foot_ripple<=.2,foot_hold_speed=foot_speed<=5.,load_balance=bool(error<=.03),foot_support=bool(abs(foot-contact)<=.01*weight),upright=bool(d['root_pos'][settled,2].min()>.7),penetration=peak_penetration>=-.005,rom=math.degrees(peak_rom)<=.1,caps=peak_cap<=.0002,finite=bool(np.isfinite(d['q']).all()))
    result=dict(model_id='human_model_v2',scenario='supported_feet',protocol=protocol,weight_n=weight,upward_harness_force_n=float(d['harness_force'][settled,0,2].mean()),expected_contact_n=float(expected),foot_contact_mean_n=float(foot),total_contact_mean_n=float(contact),load_balance_relative_error=float(error),foot_hold_peak_to_peak_deg=foot_ripple,foot_hold_speed_deg_s=foot_speed,peak_penetration_m=peak_penetration,peak_rom_excess_deg=math.degrees(peak_rom),engine_gate=dict(status='passed' if all(checks.values()) else 'candidate_failed_gate',checks=checks),shutdown='pending',scope='External pelvis harness with vertical damping and non-foot posture clamps; gravity/contact ON; plantar load plus ankle/subtalar/toe tilts, not autonomous standing')
    stem=Path(output)/f'human_model_v2_v2_suite_seed{seed}_raw';np.savez_compressed(stem.with_suffix('.npz'),**d,names=model.names,body_names=sim.robot_config.kinematic_info.body_names);stem.with_suffix('.json').write_text(json.dumps(result,indent=2));print('SUPPORTED_FEET_COMPLETE',result,flush=True)
    sim._sim.clear_all_callbacks()
    from isaaclab.sim import SimulationContext
    SimulationContext.clear_instance();app.close(wait_for_replicator=False)


def native_strength(sim,app,output,seed,protocol,initial):
    """External dynamometer drives12 limbs/directions; human torque saturates.

    Explicit fixture torque holds/moves the limb against the tested human output.
    It is not human-only movement or a standing/balance test. Gravity/contact ON,
    fixed pelvis and non-test joint clamps, feet airborne. No per-step state reset.
    """
    initial.convert_to_common(sim.data_conversion)
    m=sim._human_joint_model;device=sim.device;view=sim._robot.root_physx_view
    from ..common.profile import load_profile
    strength=load_profile('human_model_v2')['active_strength_model']
    cases=[dict(name=name,side=side,**row) for side in ('L','R') for name,row in strength['directions'].items()]
    N=len(cases);assert sim.num_envs==N
    ids=torch.arange(N,device=device);native_ids=ids.to(torch.int32);order=sim.data_conversion.dof_convert_to_common
    focus=torch.tensor([m.names.index(c['side']+'_'+c['joint']) for c in cases],device=device)
    sign=torch.tensor([c['sign'] for c in cases],device=device)
    pose=initial.dof_pos.clone();pose.zero_();initial.root_pos[:,2]=1.8
    initial.root_pos[:,0]=3+3*(ids%4);initial.root_pos[:,1]=3+3*(ids//4)
    center=[];amplitude=[]
    for n,c in enumerate(cases):
        lo,hi=c['angle_domain_rad'];i=int(focus[n]);clinical=c['clinical_angle_sign']
        bounds=sorted([float(m.lower[i])*clinical,float(m.upper[i])*clinical]);lo=max(lo,bounds[0])+.02;hi=min(hi,bounds[1])-.02
        center.append((lo+hi)/2/clinical);amplitude.append(min(.4,(hi-lo)*.45))
        side=c['side'];pose[n,m.names.index(side+'_Hip_y')]=-math.radians(80 if c['joint']=='Ankle_y' else 70 if c['joint']=='Knee_y' else 0)
        pose[n,m.names.index(side+'_Knee_y')]=math.radians(50 if c['joint']=='Ankle_y' else 10)
    center=torch.tensor(center,device=device);amp=torch.tensor(amplitude,device=device)
    pose[ids,focus]=center-amp;initial.dof_pos=pose;initial.dof_vel.zero_()
    limits=torch.stack((m.lower,m.upper),-1)[None].expand(N,-1,-1).clone();original=limits.clone()
    limits[:,:,0]=torch.maximum(m.lower,pose-math.radians(.005));limits[:,:,1]=torch.minimum(m.upper,pose+math.radians(.005));limits[ids,focus]=original[ids,focus]
    native=torch.empty_like(limits);native[:,order]=limits;view.set_dof_limits(native.cpu(),torch.arange(N,dtype=torch.int32))
    sim.reset_envs(initial);torch.testing.assert_close(sim.get_dof_state().dof_pos,pose,atol=1e-5,rtol=0)
    inertia=view.get_generalized_mass_matrices()[:,order][:,:,order].diagonal(dim1=-2,dim2=-1)+view.get_dof_armatures().to(device)[:,order]
    I=inertia[ids,focus].clamp_min(.001);kp=torch.full_like(I,2000);kd=2*torch.sqrt(kp*I)
    dt=1/sim.config.sim.fps;omega=2*math.pi*protocol.get("frequency_hz",.4);steps=round(8/dt)
    records={k:[] for k in ['time','q','qd','active','negative_cap','positive_cap','fixture_torque','native_effort','target']}
    peak_cap=peak_sum=peak_readback=peak_rom=peak_track=0.;domain_hits=0
    for step in range(steps):
        t=step*dt;state=sim.get_dof_state();q=state.dof_pos;qd=state.dof_vel
        target=center-amp*math.cos(omega*t);speed=amp*omega*math.sin(omega*t);acc=amp*omega**2*math.cos(omega*t)
        command=torch.zeros_like(q);command[ids,focus]=sign*1e4
        sim._common_actions=command;sim._apply_control();sim._scene.write_data_to_sim()
        human=sim.human_applied_torques;gravity=view.get_gravity_compensation_forces()[:,order]
        fixture=torch.zeros_like(q);fixture[ids,focus]=kp*(target-q[ids,focus])+kd*(speed-qd[ids,focus])+I*acc+gravity[ids,focus]-human[ids,focus]
        total=human+fixture;converted=torch.empty_like(total);converted[:,order]=total
        view.set_dof_actuation_forces(converted,native_ids)
        observed=view.get_dof_actuation_forces()[:,order]
        peak_readback=max(peak_readback,float((observed-total).abs().max()))
        active=sim.human_active_torques;neg=sim.human_negative_caps;pos=sim.human_positive_caps
        peak_cap=max(peak_cap,float(torch.maximum(active-pos,-active-neg).clamp_min(0).max()))
        peak_sum=max(peak_sum,float((human-active-sim.human_passive_torques).abs().max()))
        expected=torch.where(sign>0,pos[ids,focus],-neg[ids,focus]);peak_cap=max(peak_cap,float((active[ids,focus]-expected).abs().max()))
        domain_hits+=int(sim.human_strength_outside_domain[ids,focus].sum())
        if step%(sim.config.sim.fps//30)==0:
            records['time'].append(t)
            for key,val in [('q',q),('qd',qd),('active',active),('negative_cap',neg),('positive_cap',pos),('fixture_torque',fixture),('native_effort',observed),('target',target)]:records[key].append(val.cpu().numpy().copy())
        sim._sim.step(render=False);sim._scene.update(dt=dt)
        after=sim.get_dof_state();assert bool(torch.isfinite(after.dof_pos).all() and torch.isfinite(after.dof_vel).all())
        peak_rom=max(peak_rom,float(torch.maximum(m.lower-after.dof_pos,after.dof_pos-m.upper).clamp_min(0).max()))
        if t>1:peak_track=max(peak_track,float((after.dof_pos[ids,focus]-(center-amp*math.cos(omega*(t+dt)))).abs().max()))
        if step%sim.config.sim.fps==0:print('STRENGTH_SECOND',round(t),'CAP_ERROR',peak_cap,flush=True)
    checks=dict(caps=peak_cap<=.0002,torque_sum=peak_sum<=.0002,native_readback=peak_readback<=.0002,rom=math.degrees(peak_rom)<=.1,fixture_tracking=math.degrees(peak_track)<=1,domain=domain_hits==0)
    stem=Path(output)/f'human_model_v2_v2_suite_seed{seed}_raw'
    np.savez_compressed(stem.with_suffix('.npz'),**{k:np.asarray(v) for k,v in records.items()},names=m.names,focus=focus.cpu().numpy())
    result=dict(model_id='human_model_v2',scenario='strength',seed=seed,protocol=protocol,cases=cases,physics_fps=sim.config.sim.fps,cap_error_nm=peak_cap,torque_sum_error_nm=peak_sum,native_readback_error_nm=peak_readback,rom_excess_deg=math.degrees(peak_rom),fixture_tracking_max_deg=math.degrees(peak_track),domain_hits=domain_hits,engine_gate=dict(status='passed' if all(checks.values()) else 'candidate_failed_gate',checks=checks),shutdown='pending',scope=native_strength.__doc__)
    stem.with_suffix('.json').write_text(json.dumps(result,indent=2));print('VIDEO_CAPTURE_SAVED',flush=True)
    sim._sim.clear_all_callbacks()
    from isaaclab.sim import SimulationContext
    SimulationContext.clear_instance();app.close(wait_for_replicator=False)


def strength_audit(output):
    """Dense equation reproduction + held-source MVC; no fitting to holdout.

    NumPy oracle uses source coefficients independent of Torch runtime evaluation.
    Agreement with adopted curves is implementation validation, not human holdout.
    """
    from ..common.paths import PACKAGE_ROOT,WORKSPACE_ROOT
    from ..common.profile import load_profile
    from ..common.dynamics import HumanJointModel
    source=json.loads((PACKAGE_ROOT/'human_model_v1/profiles/candidate_parameters.json').read_text())['strength']
    profile=load_profile('human_model_v2');m=HumanJointModel(profile,list(profile['joints']),dtype=torch.float64)
    reports=[];traces={}
    scale=np.array([c['mass_kg']*c['height_m']*source['gravity_m_s2'] for c in source['cohorts']])
    for side in ('L','R'):
        for name,row in source['directions'].items():
            idx=m.names.index(side+'_'+row['joint']);c=np.array(row['coefficients'])
            angles=np.linspace(*row['angle_domain_rad'],41)
            speeds=np.linspace(-row['eccentric_limit_rad_s'],row['concentric_limit_rad_s'],41)
            A,V=np.meshgrid(angles,speeds,indexing='ij');a=A.reshape(-1);v=V.reshape(-1)
            iso=c[:,0]*scale*np.maximum(np.cos(c[:,1]*(a[:,None]-c[:,2])),0)
            speed=np.abs(v[:,None]);factor=(2*c[:,3]*c[:,4]+speed*(c[:,4]-3*c[:,3]))/(2*c[:,3]*c[:,4]+speed*(2*c[:,4]-4*c[:,3]))
            reference=(iso*np.maximum(factor,0)*(1+c[:,5]*np.maximum(-v[:,None],0))).mean(-1)
            q=torch.zeros((len(a),len(m.names)),dtype=torch.float64);qd=torch.zeros_like(q)
            q[:,idx]=torch.from_numpy(a/row['clinical_angle_sign']);qd[:,idx]=torch.from_numpy(v*row['sign'])
            neg,pos,_=m.strength_caps(q,qd);pred=(pos if row['sign']>0 else neg)[:,idx].numpy()
            relative=float(np.max(np.abs(pred-reference)/np.maximum(reference,1e-8)))
            requested=torch.zeros_like(q);requested[:,idx]=1e4*row['sign'];total,active,passive=m.torques(requested,q,qd)
            saturation=float(np.max(np.abs(active[:,idx].numpy()-reference*row['sign'])))
            backend=float((total.abs()-m.backend_limit).clamp_min(0).max())
            physical=(q[:,idx]>=m.lower[idx])&(q[:,idx]<=m.upper[idx])
            reports.append(dict(side=side,direction=name,points=len(a),within_model_rom_points=int(physical.sum()),source_equation_max_relative_error=relative,saturation_max_error_nm=saturation,backend_excess_nm=backend,status='passed' if relative<2e-6 and saturation<.0002 and backend<.0002 else 'candidate_failed_gate'))
            traces[side+'_'+name]=np.stack([a,v,reference,pred,physical.numpy()],axis=1)
    q=torch.zeros((1,len(m.names)),dtype=torch.float64);q[:,m.names.index('R_Knee_y')]=math.radians(20)
    cmd=q*0;idx=m.names.index('R_Ankle_y');cmd[:,idx]=1e4
    total,active,passive=m.torques(cmd,q,q*0);z=(float(total[0,idx])-100.72)/30.01
    result=dict(model_id='human_model_v2',status='passed' if all(r['status']=='passed' for r in reports) and abs(z)<=2 else 'candidate_failed_gate',directions=reports,
        independent_wang2021=dict(active_nm=float(active[0,idx]),passive_nm=float(passive[0,idx]),total_nm=float(total[0,idx]),mean_nm=100.72,sd_nm=30.01,z=z,status='within_2sd_proxy' if abs(z)<=2 else 'outside_2sd_proxy',source='https://doi.org/10.1155/2021/8899699',scope='Knee20 ankle0, independent-source scale only; source dynamometer axis differs from model generalized axis'),
        scope='12 direction equation grids, including flagged positions outside model ROM for algebra only. No source-curve agreement claimed as independent biological validation.',
        outside_domain_policy=source['outside_domain'])
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/'strength_equation_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(output/'strength_equation_audit.npz',**traces)
    return result


def strength_trace_audit(report_path):
    """Compare measured native active output to independent NumPy source curves."""
    from ..common.paths import PACKAGE_ROOT
    report_path=Path(report_path);report=json.loads(report_path.read_text())
    trace=np.load(report_path.with_suffix('.npz'));source=json.loads((PACKAGE_ROOT/'human_model_v1/profiles/candidate_parameters.json').read_text())['strength']
    scales=np.array([c['mass_kg']*c['height_m']*source['gravity_m_s2'] for c in source['cohorts']]);rows=[]
    for env,case in enumerate(report['cases']):
        i=int(trace['focus'][env]);row=source['directions'][case['name']];c=np.array(row['coefficients'])
        a=trace['q'][:,env,i].astype(float)*row['clinical_angle_sign'];v=trace['qd'][:,env,i].astype(float)*row['sign']
        a=np.clip(a,*row['angle_domain_rad']);v=np.clip(v,-row['eccentric_limit_rad_s'],row['concentric_limit_rad_s'])
        speed=np.abs(v[:,None]);iso=c[:,0]*scales*np.maximum(np.cos(c[:,1]*(a[:,None]-c[:,2])),0)
        factor=(2*c[:,3]*c[:,4]+speed*(c[:,4]-3*c[:,3]))/(2*c[:,3]*c[:,4]+speed*(2*c[:,4]-4*c[:,3]))
        predicted=(iso*np.maximum(factor,0)*(1+c[:,5]*np.maximum(-v[:,None],0))).mean(-1)*row['sign']
        observed=trace['active'][:,env,i];error=float(np.max(np.abs(observed-predicted)));relative=float(np.max(np.abs(observed-predicted)/np.maximum(np.abs(predicted),1e-8)))
        rows.append(dict(side=case['side'],direction=case['name'],max_error_nm=error,max_relative_error=relative,active_min_nm=float(observed.min()),active_max_nm=float(observed.max()),samples=len(observed),status='passed' if error<=.0002 and relative<=2e-6 else 'candidate_failed_gate'))
    result=dict(status='passed' if all(r['status']=='passed' for r in rows) else 'candidate_failed_gate',source_report=str(report_path),rows=rows,scope='Native physics q/qd and active actuator output vs adopted literature equation; fixture-assisted motion, not independent human validation')
    report_path.with_name(report_path.stem.replace('_raw','_literature_trace')+'.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def rom_capacity_audit(report_path):
    """Explain legacy active-to-passive-endpoint failures without changing gates.

    Independent MJCF gravity and runtime passive/cap functions at recorded hold
    pose and desired endpoint. This is force-budget diagnosis, not a human test.
    """
    import mujoco
    from ..common.paths import PACKAGE_ROOT
    from ..common.profile import load_profile
    from ..common.dynamics import HumanJointModel
    p=Path(report_path);report=json.loads(p.read_text());trace=np.load(p.with_suffix('.npz'));names=list(trace['names'])
    profile=load_profile('human_model_v2');force=HumanJointModel(profile,names,dtype=torch.float64)
    model=mujoco.MjModel.from_xml_path(str(PACKAGE_ROOT/'human_model_v2/assets/human_model_v2.xml'));data=mujoco.MjData(model)
    ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names]
    qp=model.jnt_qposadr[ids];dv=model.jnt_dofadr[ids];rows=[]
    def budget(q,i):
        data.qpos[:]=model.qpos0;data.qpos[qp]=q;data.qvel[:]=0;mujoco.mj_forward(model,data)
        qt=torch.tensor(q[None],dtype=torch.float64);passive=float(force.passive_torque(qt,qt*0)[0,i]);neg,pos,_=force.strength_caps(qt,qt*0)
        gravity=float(data.qfrc_bias[dv[i]]);required=gravity-passive
        deficit=max(0,required-float(pos[0,i]),-required-float(neg[0,i]))
        return dict(gravity_nm=gravity,passive_nm=passive,required_active_nm=required,negative_cap_nm=float(neg[0,i]),positive_cap_nm=float(pos[0,i]),deficit_nm=deficit)
    t=trace['time'];hold=(t>=7.6)&(t<8.4)
    for env,row in enumerate(report['direction_results']):
        if row['status']=='passed':continue
        i=names.index(row['joint']);q=trace['q'][hold,env].mean(0).astype(float);target=q.copy();target[i]=float(trace['target'][hold,env].mean())
        actual=budget(q,i);desired=budget(target,i);active=float(trace['active'][hold,env,i].mean())
        item=dict(joint=row['joint'],direction=row['direction'],held_angle_deg=float(np.rad2deg(q[i])),target_angle_deg=float(np.rad2deg(target[i])),actual=actual,target=desired,held_active_nm=active,held_force_balance_error_nm=abs(active-actual['required_active_nm']),original_checks=row['checks'])
        if not row['checks']['recovery']:
            returnq=trace['q'][t>18,env].mean(0).astype(float);desiredreturn=returnq.copy();desiredreturn[i]=0
            item['return_angle_deg']=float(np.rad2deg(returnq[i]));item['return_target_budget']=budget(desiredreturn,i)
        rows.append(item)
    result=dict(scope='Legacy gate expects active access to passive limits. Saturation/force deficits are retained, not repaired by increasing strength or relabelled as full ROM success.',rows=rows)
    p.with_name(p.stem.replace('_raw','_capacity_diagnosis')+'.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def population_strength_scale_audit(output):
    """Broad morphology-matched scale checks, not exact mean fitting.

    Cohort means within15% mass and5% height are preferred references, a project
    screening convention. No individual anthropometric matching is claimed.
    Body-normalized values are descriptive, not subject-level normalized SD.
    """
    from ..common.paths import WORKSPACE_ROOT
    from ..common.profile import load_profile
    from ..common.dynamics import HumanJointModel
    src=WORKSPACE_ROOT/'output/260914/e01_human_model_population/sources'
    registry=json.loads((src/'population_reference_registry.json').read_text());mass=registry['model_mass_kg'];height=registry['model_height_proxy_m']
    profile=load_profile('human_model_v2');m=HumanJointModel(profile,list(profile['joints']),dtype=torch.float64)
    morin=json.loads((src/'morin2023_population.json').read_text());rows=[]
    for source,records,body in [('morin2023',morin['records'],{'mass_kg':71.8,'height_m':1.73}),('wang2021',[dict(joint='R_Ankle_y',direction=1,label='Ankle plantarflexion',mean_nm=100.72,sd_nm=30.01,n=10)],{'mass_kg':66.7,'height_m':1.718})]:
        for ref in records:
            if ref['joint'] not in ('R_Knee_y','R_Ankle_y'):continue
            q=torch.zeros((1,len(m.names)),dtype=torch.float64);qd=q*0
            if ref['joint']=='R_Knee_y':q[:,m.names.index('R_Hip_y')]=-math.pi/2;q[:,m.names.index('R_Knee_y')]=math.pi/2
            elif source=='wang2021':q[:,m.names.index('R_Knee_y')]=math.radians(20)
            # Morin supports the knee with a pillow: angle is unspecified.
            # Use neutral proxy and explicitly retain this protocol limitation.
            i=m.names.index(ref['joint']);request=q*0;request[:,i]=ref['direction']*1e4
            total,active,passive=m.torques(request,q,qd);value=float(active[0,i])*ref['direction'];net=float(total[0,i])*ref['direction']
            az=(value-ref['mean_nm'])/ref['sd_nm'];nz=(net-ref['mean_nm'])/ref['sd_nm']
            size_match=abs(body['mass_kg']/mass-1)<=.15 and abs(body['height_m']/height-1)<=.05
            rows.append(dict(source=source,label=ref['label'],joint=ref['joint'],direction=ref['direction'],n=ref['n'],model_active_nm=value,model_net_nm=net,mean_nm=ref['mean_nm'],sd_nm=ref['sd_nm'],active_z=az,net_z=nz,model_active_nm_per_kg=value/mass,reference_mean_nm_per_kg=ref['mean_nm']/body['mass_kg'],model_active_over_mgh=value/(mass*9.81*height),reference_mean_over_mgh=ref['mean_nm']/(body['mass_kg']*9.81*body['height_m']),cohort_body=body,preferred_body_match=size_match,status='within_2sd_scale_proxy' if size_match and max(abs(az),abs(nz))<=2 else 'review_required',scope='Cohort mean morphology, apparatus-axis and passive/gravity correction mismatch retained; Morin knee support angle unspecified for ankle assay. Morin was used by historical static baseline, not a wholly untouched dataset. No tuning to these values.'))
    result=dict(status='plausible_scale_with_limits' if all(r['status']=='within_2sd_scale_proxy' for r in rows) else 'review_required',model_body=dict(mass_kg=mass,height_proxy_m=height),criteria=dict(mass_fraction=.15,height_fraction=.05,absolute_z_max=2,interpretation='Broad descriptive project criteria, not clinical standards; preserve variation rather than fit means'),rows=rows,limitations=['No individual body-linked torque observations or subject-matched inertias available.','Nm/kg and Nm/(mgh) use cohort mean denominators and are descriptive, not independent normalized distribution estimates.','Isometric morphology matching does not establish gait validity or active ROM under arbitrary fixture loads.','Two literature populations, not repeated sessions counted as extra independent people.'])
    output=Path(output);output.mkdir(parents=True,exist_ok=True);(output/'population_strength_scale.json').write_text(json.dumps(result,indent=2)+'\n');return result


def native_retarget_tracking(sim, app, output, seed, protocol, initial):
    """Diagnostic PD tracking with an explicit moving pelvis force harness.

    Gravity/contact on, no joint teleport/clamps, fixed strength profile. This
    does not establish self-supported gait or policy performance.
    """
    from .motion_lib import HumanModelMotionLib
    from protomotions.components.motion_lib import MotionLibConfig
    from protomotions.utils.rotations import quat_mul, quat_conjugate
    device=sim.device;model=sim._human_joint_model;fps=sim.config.sim.fps;dt=1/fps
    lib=HumanModelMotionLib(MotionLibConfig(motion_file=protocol['motion_file']),str(device))
    ids=torch.tensor(protocol['motion_ids'],device=device,dtype=torch.long)
    time_scales=torch.tensor(protocol.get('time_scales',[1.]*len(ids)),device=device)
    if time_scales.shape!=ids.shape or not bool(torch.isfinite(time_scales).all() and (time_scales>0).all()):
        raise ValueError('Positive finite time scale per motion required')
    assert len(ids)==sim.num_envs and not sim._robot.is_fixed_base
    initial.convert_to_common(sim.data_conversion)
    initial_ref=lib.get_motion_state(ids,torch.zeros(len(ids),device=device))
    initial_ref.dof_vel*=time_scales[:,None]
    initial_ref.rigid_body_vel*=time_scales[:,None,None]
    initial_ref.rigid_body_ang_vel*=time_scales[:,None,None]
    offset=torch.zeros(len(ids),3,device=device)
    spawn_xy=torch.stack((10.+10.*torch.arange(len(ids),device=device),torch.full((len(ids),),10.,device=device)),-1)
    offset[:,:2]=spawn_xy-initial_ref.rigid_body_pos[:,0,:2]
    offset[:,2]=protocol.get('root_clearance_m',0.)
    initial.root_pos[:]=initial_ref.rigid_body_pos[:,0]+offset
    initial.root_rot[:]=initial_ref.rigid_body_rot[:,0]
    initial.root_vel[:]=initial_ref.rigid_body_vel[:,0]
    initial.root_ang_vel[:]=initial_ref.rigid_body_ang_vel[:,0]
    initial.dof_pos[:]=initial_ref.dof_pos;initial.dof_vel[:]=initial_ref.dof_vel
    sim.reset_envs(initial)
    pelvis=sim._robot.body_names.index('Pelvis')
    masses=sim._robot.root_physx_view.get_masses().to(device).sum(-1)
    duration=min(float((lib.motion_lengths[ids]/time_scales).min()),protocol.get('duration_s',6.))
    records={k:[] for k in ('time','q','qd','q_ref','qdot_ref','requested','active','passive','applied','negative_cap','positive_cap','root_pos','root_ref','harness_force','harness_torque','contact_forces')}
    saturation=torch.zeros(len(ids),59,device=device);squared=torch.zeros_like(saturation)
    maximum=torch.zeros_like(saturation);peak_cap=0.;peak_rom=0.;peak_sum=0.
    peak_fk_error=0.;minimum_foot_z=float('inf')
    from .retarget import forward
    from .retarget_metrics import box_corners, world_points, FOOT_NAMES
    from ..common.paths import PACKAGE_ROOT
    from protomotions.utils.rotations import quaternion_to_matrix
    foot_ids=[lib.kinematics.body_names.index(n) for n in FOOT_NAMES]
    foot_corners=box_corners(PACKAGE_ROOT/'human_model_v2/assets/human_model_v2.xml',FOOT_NAMES,str(device))
    steps=round(duration*fps)
    # Precompute reference FK once, rather than thousands of Python/GPU FK
    # launches in the physics loop. The dynamics still runs every substep.
    query_ids=ids.repeat(steps)
    query_times=torch.arange(steps,device=device).repeat_interleave(len(ids))/fps
    query_times*=time_scales.repeat(steps)
    trajectory=lib.get_motion_state(query_ids,query_times)
    trajectory.dof_vel*=time_scales.repeat(steps)[:,None]
    trajectory.rigid_body_vel*=time_scales.repeat(steps)[:,None,None]
    trajectory.rigid_body_ang_vel*=time_scales.repeat(steps)[:,None,None]
    order=sim.data_conversion.dof_convert_to_common
    view=sim._robot.root_physx_view
    training_pd=protocol.get('pd_mode')=='training_pd'
    if training_pd:
        # Match the teacher's q-target PD (zero desired velocity), with only
        # trunk gains varied. The pelvis harness remains explicitly external.
        settings=protocol['pd_settings']
        if len(settings)!=len(ids):raise ValueError('One PD setting per environment required')
        kp=torch.maximum(model.negative,model.positive).expand(len(ids),-1).clone()/1.5
        kp=torch.where(kp>0,kp,torch.full_like(kp,500.))
        kd=.1*kp
        trunk=[model.names.index(f'{body}_{axis}') for body in ('Torso','Spine','Chest') for axis in 'xyz']
        for env,setting in enumerate(settings):
            kp[env,trunk]*=setting['trunk_kp_multiplier']
            kd[env,trunk]*=setting['trunk_kd_multiplier']
        target=initial_ref.dof_pos.clone()
    for step in range(steps):
        t=step*dt
        from types import SimpleNamespace
        ix=slice(step*len(ids),(step+1)*len(ids))
        reference=SimpleNamespace(**{k:getattr(trajectory,k)[ix] for k in
            ('dof_pos','dof_vel','rigid_body_pos','rigid_body_rot','rigid_body_vel','rigid_body_ang_vel')})
        state=sim.get_dof_state();root=sim.get_root_state()
        if training_pd:
            if step%(fps//30)==0:target=reference.dof_pos.clone()
            command=kp*(target-state.dof_pos)-kd*state.dof_vel
        elif protocol.get('pd_mode','fixed') in ('inertia_scaled','critical'):
            inertia=view.get_generalized_mass_matrices()[:,6:,6:][:,order][:,:,order].diagonal(dim1=-2,dim2=-1)
            inertia=inertia+view.get_dof_armatures().to(device)[:,order]
            if protocol.get('pd_mode')=='critical':
                kp=protocol.get('joint_kp',500.)
                kd=2*torch.sqrt(kp*inertia.clamp_min(.001))
                command=kp*(reference.dof_pos-state.dof_pos)+kd*(reference.dof_vel-state.dof_vel)
            else:
                command=inertia*(200*(reference.dof_pos-state.dof_pos)+28.284*(reference.dof_vel-state.dof_vel))
            gravity=joint_generalized_forces(view.get_gravity_compensation_forces(),order,floating_base=True)
            command+=gravity-model.passive_torque(state.dof_pos,state.dof_vel)
        else:
            command=500*(reference.dof_pos-state.dof_pos)+50*(reference.dof_vel-state.dof_vel)
        root_ref=reference.rigid_body_pos[:,0]+offset
        force=4000*(root_ref-root.root_pos)+400*(reference.rigid_body_vel[:,0]-root.root_vel)
        force[:,2]+=.5*masses*9.81
        error=quat_mul(reference.rigid_body_rot[:,0],quat_conjugate(root.root_rot,True),True)
        torque=protocol.get('harness_rotation_kp',100.)*2*error[:,:3]*torch.sign(error[:,3:])+protocol.get('harness_rotation_kd',5.)*(reference.rigid_body_ang_vel[:,0]-root.root_ang_vel)
        torque=torque.clamp(-protocol.get('harness_torque_limit_nm',200.),protocol.get('harness_torque_limit_nm',200.))
        sim._robot.set_external_force_and_torque(force[:,None],torque[:,None],body_ids=[pelvis],is_global=True)
        sim._common_actions=command;sim._apply_control();sim._scene.write_data_to_sim()
        saturation+=(command>sim.human_positive_caps+1e-5)|(command < -sim.human_negative_caps-1e-5)
        peak_cap=max(peak_cap,float(torch.maximum(sim.human_active_torques-sim.human_positive_caps,-sim.human_active_torques-sim.human_negative_caps).clamp_min(0).max()))
        peak_sum=max(peak_sum,float((sim.human_applied_torques-sim.human_active_torques-sim.human_passive_torques).abs().max()))
        sim._sim.step(render=False);sim._scene.update(dt=dt)
        after=sim.get_dof_state()
        if not bool(torch.isfinite(after.dof_pos).all() and torch.isfinite(after.dof_vel).all()):
            raise RuntimeError('Non-finite native retarget tracking state')
        err=(after.dof_pos-reference.dof_pos).abs();squared+=err.square();maximum=torch.maximum(maximum,err)
        peak_rom=max(peak_rom,float(torch.maximum(model.lower-after.dof_pos,after.dof_pos-model.upper).clamp_min(0).max()))
        if step%(fps//30)==0:
            native=sim.get_robot_state()
            predicted,_,_=forward(lib.kinematics,native.dof_pos,native.rigid_body_pos[:,0],quaternion_to_matrix(native.rigid_body_rot[:,0],w_last=True))
            fk_error=float((predicted-native.rigid_body_pos).norm(dim=-1).max())
            peak_fk_error=max(peak_fk_error,fk_error)
            bottom=float(world_points(native.rigid_body_pos,quaternion_to_matrix(native.rigid_body_rot,w_last=True),foot_ids,foot_corners)[...,2].min())
            minimum_foot_z=min(minimum_foot_z,bottom)
            if step<fps:print('SMPL2HM_FK_CONTACT',round(t,3),'FK_M',fk_error,'FOOT_Z',bottom,flush=True)
            records['time'].append(t)
            for key,value in [('q',after.dof_pos),('qd',after.dof_vel),('q_ref',reference.dof_pos),('qdot_ref',reference.dof_vel),('requested',command),('active',sim.human_active_torques),('passive',sim.human_passive_torques),('applied',sim.human_applied_torques),('negative_cap',sim.human_negative_caps),('positive_cap',sim.human_positive_caps),('root_pos',sim.get_root_state().root_pos),('root_ref',root_ref),('harness_force',force),('harness_torque',torque),('contact_forces',sim.get_bodies_contact_buf().rigid_body_contact_forces)]:
                records[key].append(value.detach().cpu().numpy().copy())
        if step%fps==0:print('SMPL2HM_TRACKING_SECOND',round(t),'MAX_ERROR_DEG',float(torch.rad2deg(maximum).max()),flush=True)
    stem=Path(output)/f'human_model_v2_v2_suite_seed{seed}_raw'
    np.savez_compressed(stem.with_suffix('.npz'),**{k:np.asarray(v) for k,v in records.items()},names=model.names,body_names=sim.robot_config.kinematic_info.body_names)
    if training_pd:
        protocol=dict(protocol,measured_kp_nm_per_rad=kp.cpu().tolist(),measured_kd_nms_per_rad=kd.cpu().tolist())
    result=dict(scenario='smpl2hm_tracking',protocol=protocol,duration_s=duration,physics_fps=fps,
        rmse_deg=torch.rad2deg((squared/steps).sqrt()).cpu().tolist(),max_error_deg=torch.rad2deg(maximum).cpu().tolist(),
        saturation_fraction=(saturation/steps).cpu().tolist(),peak_cap_excess_nm=peak_cap,peak_rom_excess_deg=math.degrees(peak_rom),
        torque_sum_error_nm=peak_sum,native_fk_max_error_m=peak_fk_error,minimum_foot_geometry_z_m=minimum_foot_z,physical_tracking_status='diagnostic_no_performance_gate',
        engine_gate=dict(status='passed' if peak_cap<.0002 and peak_sum<.0002 and math.degrees(peak_rom)<=.1 else 'candidate_failed_gate'),
        shutdown='pending',scope='Moving pelvis force/torque harness; gravity/contact ON; not autonomous walking; '+('30 Hz held q-target, zero desired velocity' if training_pd else 'joint PD with qdot_ref feedforward'))
    stem.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    print('SMPL2HM_TRACKING_SAVED',flush=True)
    sim._sim.clear_all_callbacks()
    from isaaclab.sim import SimulationContext
    SimulationContext.clear_instance();app.close(wait_for_replicator=False)


def joint_generalized_forces(forces, order, *, floating_base):
    """Strip the six floating-base wrench entries before indexing joint forces."""
    offset=6 if floating_base else 0
    if forces.shape[-1] != len(order)+offset:
        raise ValueError('Unexpected generalized force dimension')
    return forces[:,offset:][:,order]


def trunk_inverse_dynamics(motion_file, *, human_model_parameters=None, time_scales=(1.,)):
    """Prescribed-motion capacity diagnostic, NOT a free-gait feasibility test.

    Pelvis wrench supplies support; contacts are disabled. For trunk joints,
    this measures the generalized moment needed to move the prescribed upper
    body, assuming no external hand/head forces. Finite-difference endpoints
    are excluded. Retiming is a synthetic stress condition, not human data.
    """
    import mujoco
    from ..common.paths import PACKAGE_ROOT
    from ..common.profile import load_profile
    from ..common.dynamics import HumanJointModel
    pack=torch.load(motion_file,map_location='cpu',weights_only=False)
    names=pack['human_model_metadata']['dof_names']
    plant=HumanJointModel(load_profile('human_model_v2'),names,dtype=torch.float64,**(human_model_parameters or {}))
    mj=mujoco.MjModel.from_xml_path(str(PACKAGE_ROOT/'human_model_v2/assets/human_model_v2.xml'))
    mj.opt.disableflags|=mujoco.mjtDisableBit.mjDSBL_CONTACT
    data=mujoco.MjData(mj)
    joint_ids=[mujoco.mj_name2id(mj,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names]
    if min(joint_ids)<0:raise ValueError('Missing MJCF joint')
    qp=mj.jnt_qposadr[joint_ids];qv=mj.jnt_dofadr[joint_ids]
    trunk=[i for i,n in enumerate(names) if n.split('_')[0] in ('Torso','Spine','Chest')]
    rows=[]
    for scale in time_scales:
        if not math.isfinite(scale) or scale<=0:raise ValueError('Positive time scales required')
        for mid,(start,count) in enumerate(zip(pack['length_starts'],pack['motion_num_frames'])):
            start=int(start);count=int(count);dt=float(pack['motion_dt'][mid])/scale
            sl=slice(start,start+count);q=np.repeat(mj.qpos0[None],count,axis=0)
            q[:,:3]=pack['gts'][sl,0].numpy();rot=pack['grs'][sl,0].numpy()
            q[:,3:7]=rot[:,[3,0,1,2]];q[:,qp]=pack['dps'][sl].numpy()
            velocity=np.zeros((count,mj.nv))
            for t in range(1,count-1):mujoco.mj_differentiatePos(mj,velocity[t],2*dt,q[t-1],q[t+1])
            acceleration=np.gradient(velocity,dt,axis=0)
            torque=[]
            for t in range(2,count-2):
                data.qpos[:]=q[t];data.qvel[:]=velocity[t];data.qacc[:]=acceleration[t]
                mujoco.mj_inverse(mj,data);torque.append(data.qfrc_inverse[qv].copy())
            pos=torch.tensor(q[2:-2,qp]);vel=torch.tensor(velocity[2:-2,qv]);tau=torch.tensor(np.asarray(torque))
            passive=plant.passive_torque(pos,vel);active=tau-passive
            neg,positive,_=plant.strength_caps(pos,vel);caps=torch.where(active>=0,positive,neg)
            ratio=active.abs()/caps.clamp_min(1e-9)
            speed=float(pack['gvs'][sl,0,:2].norm(dim=-1).mean())*scale
            rows.append(dict(motion_id=mid,time_scale=scale,mean_speed_m_s=speed,
                source=str(pack['motion_files'][mid]),trunk={names[i]:dict(
                    active_required_peak_nm=float(active[:,i].abs().max()),active_required_p95_nm=float(torch.quantile(active[:,i].abs(),.95)),
                    capacity_ratio_peak=float(ratio[:,i].max()),capacity_ratio_p95=float(torch.quantile(ratio[:,i],.95)),
                    capacity_exceeded_fraction=float((ratio[:,i]>1).double().mean()),
                    passive_peak_nm=float(passive[:,i].abs().max())) for i in trunk}))
    return dict(status='diagnostic',motion_file=str(motion_file),human_model_parameters=human_model_parameters or {},
        scope='Prescribed pelvis wrench, no contact; static trunk capacity proxy, no trunk force-velocity model. Source retarget acceleration can contain artifacts. Not autonomous gait or vigorous human movement validation.',rows=rows)
