import numpy as np
import pytest


def test_all_direction_rom_coverage_and_smooth_return():
    from protomotions.robot_configs.human_model.validation import all_direction_rom_protocols,population_fixture_target
    from protomotions.robot_configs.human_model.profile import load_profile
    profile=load_profile();protocols=all_direction_rom_protocols(profile)
    chapters=[c for p in protocols.values() for c in p['chapters']]
    assert len(chapters)==138
    assert {(c['joint'],c['direction']) for c in chapters}=={(j,s) for j in profile['joints'] for s in (-1,1)}
    for c in chapters:
        knots=c['trajectory_knots']
        assert all(b[0]>a[0] for a,b in zip(knots,knots[1:]))
        for t,q in knots:
            actual,velocity,acceleration=population_fixture_target(c,t)
            assert np.rad2deg(actual)==pytest.approx(q,abs=1e-8)
            assert velocity==pytest.approx(0,abs=1e-8)
            assert acceleration==pytest.approx(0,abs=1e-8)
        assert knots[-1][1]==c['start_deg']
        assert c['external_probe_nm']==(.5 if max(profile['joints'][c['joint']]['active_nm'])==0 else 0)


def test_rom_suite_rejects_hidden_assistance_and_unreached_limit():
    from protomotions.robot_configs.human_model.metrics import all_direction_rom_metrics
    t=np.arange(.5,8.1,.5);q=np.zeros((len(t),2));q[(t>1)&(t<=5),0]=np.deg2rad(10)
    zero=np.zeros_like(q)
    trace=dict(dof_names=['J_x','K_x'],chapter_index=np.zeros(len(t)),time=t,q=q,target=q.copy(),
        lower=np.deg2rad([-10,-10]),upper=np.deg2rad([10,10]),contact_points_count=np.zeros(len(t)),
        contact_min_separation_m=np.zeros(len(t)),inertia_accel=zero,coriolis=zero,gravity=zero,engine_total=zero)
    trace['engine_total']=zero.copy();trace['engine_total'][(t>4)&(t<=5),0]=1
    sub=dict(q=q,qd=zero,active=zero,requested=zero,positive_cap=zero+100,negative_cap=zero+100,
        applied=zero,elastic=zero,damping=zero)
    audit=dict(chapter_index=np.zeros(len(t)),fixture_focus_nm=np.zeros(len(t)))
    c=dict(joint='J_x',direction=1,label='test',start_s=0,start_deg=0,end_deg=10,locked_axis=False,
        endpoint_window_s=[1,2],recovery_window_s=[7,8],probe_window_s=[4,5],rom_deg=[-10,10])
    thresholds=dict(endpoint_error_deg=2,recovery_error_deg=2,hold_peak_to_peak_deg=1,rom_excess_deg=1,
        non_test_error_deg=2,cap_error_nm=.0002,torque_sum_error_nm=.0002)
    assert all_direction_rom_metrics(trace,sub,audit,[c],thresholds)[0]['status']=='passed'
    audit['fixture_focus_nm'][3]=.1
    assert not all_direction_rom_metrics(trace,sub,audit,[c],thresholds)[0]['checks']['no_external_assistance']
    audit['fixture_focus_nm']*=0
    q[(t>4)&(t<=5),0]=0
    row=all_direction_rom_metrics(trace,sub,audit,[c],thresholds)[0]
    assert row['limit_test_status']=='not_exercised'
    assert not row['checks']['boundary_exercised']


def test_native_clock_grid_keeps_long_run_quantization_separate_from_missing_steps():
    from protomotions.robot_configs.human_model.metrics import native_clock_grid_metrics
    n=400000;dt=float(np.float32(1/480));time=np.arange(n)*dt
    row=native_clock_grid_metrics(time,physics_fps=480,expected_count=n,terminal_time=n*dt)
    assert row['passed']
    assert not row['nominal_grid']['passed']
    time[n//2]=time[n//2-1]
    assert not native_clock_grid_metrics(time,physics_fps=480,expected_count=n,terminal_time=n*dt)['passed']


def test_fixed_base_inertia_audit_maps_names_and_separates_armature(tmp_path):
    import mujoco
    from protomotions.robot_configs.human_model.validation import fixed_base_inertia_audit
    source=tmp_path/'two_hinges.xml'
    source.write_text('<mujoco><worldbody><body><joint name="a" axis="1 0 0" armature=".02"/>'
                      '<geom type="sphere" size=".1" mass="1"/><body pos="0 0 .3">'
                      '<joint name="b" axis="0 1 0" armature=".03"/>'
                      '<geom type="sphere" size=".1" mass="2"/></body></body></worldbody></mujoco>')
    model=mujoco.MjModel.from_xml_path(str(source));state=mujoco.MjData(model)
    state.qpos[:]=[.2,.4];mujoco.mj_forward(model,state)
    matrix=np.empty((2,2));mujoco.mj_fullM(model,state,matrix)
    report=fixed_base_inertia_audit(matrix[::-1,::-1],['b','a'],[.4,.2],source)
    assert report['comparisons']['authored_armature']['max_abs_kg_m2']==pytest.approx(0,abs=1e-14)
    assert report['comparisons']['zero_armature']['max_abs_kg_m2']==pytest.approx(.03)
    assert report['authored_armature_kg_m2']==[.03,.02]
    assert report['minimum_symmetric_eigenvalue_kg_m2']>0
    for names in (['a','a'],['a','missing']):
        with pytest.raises(ValueError):
            fixed_base_inertia_audit(matrix,names,[.2,.4],source)


def test_video_gate_rejects_missing_exercise_reset_and_substep_rom_failure():
    from protomotions.robot_configs.human_model.validation import video_engine_checks
    phase=np.repeat(['hold','joint sweep','passive release','torque saturation','reset and hold'],[60,180,120,60,60])
    q=np.zeros((480,2));q[60:240,0]=np.linspace(0,.8,180)
    active=np.zeros_like(q);active[360:390,0]=1;active[390:420,0]=-1
    data=dict(time=np.arange(1,481)/30,phase=phase,q=q,qd=np.zeros_like(q),
              active=active,requested=active*300,applied=active,engine=active.copy())
    metrics=dict(rom_excess_max_deg=[0,0],cap_violation_max_nm=[0,0],torque_accounting_max_nm=[0,0],
                 damping_energy_injection_j=[0,0],n_substeps=7680,saturation_fraction=[.1,0],sampling_grid={'passed':True})
    reset=[dict(q_error_rad=0,qd_error_rad_s=0)]
    assert video_engine_checks(data,metrics,reset,True)['status']=='passed'
    assert video_engine_checks(data,metrics,[],True)['checks']['reset']['passed'] is False
    bad=dict(metrics,rom_excess_max_deg=[np.rad2deg(1.01e-4),0])
    # Frame samples remain in ROM: a failure between frames must still fail.
    assert video_engine_checks(data,bad,reset,True)['checks']['rom_excess_rad']['passed'] is False
    idle=dict(data,q=np.zeros_like(q),requested=np.zeros_like(q),active=np.zeros_like(q))
    checks=video_engine_checks(idle,metrics,reset,True)['checks']
    assert not checks['motion_exercised']['passed'] and not checks['saturation_exercised']['passed']
    reordered=dict(data,phase=np.roll(phase,1))
    assert not video_engine_checks(reordered,metrics,reset,True)['checks']['complete_phases']['passed']
    missing_grid=dict(metrics);missing_grid.pop('sampling_grid')
    assert not video_engine_checks(data,missing_grid,reset,True)['checks']['physics_sampling']['passed']


def test_sampling_grid_detects_equal_count_corruption_and_wrong_endpoint():
    from protomotions.robot_configs.human_model.metrics import sampling_grid_metrics
    t=np.arange(100)/480
    args=dict(expected_dt=1/480,expected_count=100,terminal_time=100/480)
    assert sampling_grid_metrics(t,**args)['passed']
    # Same number of samples, but one physical sample is replaced by a duplicate.
    duplicate=t.copy();duplicate[50]=duplicate[49]
    assert not sampling_grid_metrics(duplicate,**args)['passed']
    delayed=t.copy();delayed[50:]+=1/480
    assert not sampling_grid_metrics(delayed,**args)['passed']
    assert not sampling_grid_metrics(t[1:],**args)['passed']
    assert not sampling_grid_metrics(t,**dict(args,terminal_time=t[-1]))['passed']
    jitter=t+1e-8*np.sin(np.arange(100))
    assert sampling_grid_metrics(jitter,**args)['passed']

from protomotions.robot_configs.human_model.metrics import (
    curve_metrics, gate, impedance_fit, step_response_metrics, torque_trace_metrics,
    rom_boundary_metrics, contact_metrics, cycle_work, frequency_response_metrics,
)


def test_recovery_time_uses_external_deficit_and_preserves_censoring():
    from protomotions.robot_configs.human_model.metrics import recovery_time_metrics
    time = np.linspace(0, 30, 3001)
    capacity = 100-40*np.exp(-time/10)
    result = recovery_time_metrics(time, capacity, initial_capacity=60, baseline_capacity=100)
    assert result['threshold_capacity'] == 96
    assert result['recovery_time_s'] == pytest.approx(10*np.log(10), abs=2e-6)
    truncated = recovery_time_metrics(time[:1001], capacity[:1001], initial_capacity=60, baseline_capacity=100)
    assert truncated['recovery_time_s'] is None and truncated['censoring'] == 'right'
    early = recovery_time_metrics([0, 1], [97, 99], initial_capacity=60, baseline_capacity=100)
    assert early['recovery_time_s'] is None and early['censoring'] == 'left'
    for initial,baseline,fraction in [(100,100,.9),(101,100,.9),(-1,100,.9),(60,100,1),(60,np.nan,.9)]:
        with pytest.raises(ValueError):
            recovery_time_metrics(time,capacity,initial_capacity=initial,baseline_capacity=baseline,fraction=fraction)


def test_endurance_capacity_crossing_and_censoring():
    from protomotions.robot_configs.human_model.metrics import endurance_time_metrics
    result = endurance_time_metrics([0, 2, 4], [1, .8, .4], target=.6)
    assert result['endurance_time_s'] == pytest.approx(3)
    assert result['crossing_bracket_s'] == [2, 4]
    assert result['censoring'] == 'none'
    # Equality alone is not failure; finite observation is not infinite ET.
    result = endurance_time_metrics([0, 2], [1, .6], target=.6)
    assert result['endurance_time_s'] is None and result['censoring'] == 'right'
    result = endurance_time_metrics([0, 2], [.5, .4], target=.6)
    assert result['endurance_time_s'] is None and result['censoring'] == 'left'
    for t,c,target in [([1,2],[1,.5],.6), ([0,2],[1,-1],.6),
                       ([0,2],[[1,.5]],.6), ([0,2],[1,.5],0)]:
        with pytest.raises(ValueError):
            endurance_time_metrics(t,c,target=target)


def test_error_metrics_reject_missing_and_broadcast_and_constant_r2():
    for p, r in (([], []), ([np.nan], [1]), ([1], [[1]])):
        with pytest.raises(ValueError):
            curve_metrics(p, r, scale=1)
    result = curve_metrics([2, 2], [1, 1], scale=4, uncertainty=[.5, .5])
    assert result["rmse"] == 1 and result["nrmse"] == .25 and result["r2"] is None
    assert result["standardized_rmse"] == 2
    assert gate(0, .01, evidence_kind="synthetic")["status"] == "insufficient_evidence"
    assert gate(None, .01, evidence_kind="independent_measurement")["status"] == "not_evaluated"


def test_contract_gate_tracks_threshold_and_does_not_promote_screening(tmp_path):
    import json
    from protomotions.robot_configs.human_model.metrics import contract_gate
    path = tmp_path/'contract.json'
    row = {'id':'test', 'threshold_max':.15, 'required_evidence':'independent_measurement',
           'threshold_basis':'engineering_acceptance_target'}
    path.write_text(json.dumps({'metrics':[row]}))
    first = contract_gate(.12,'test',evidence_kind='digitized_measurement',
                          required_evidence='digitized_measurement',contract_path=path)
    assert first['status'] == 'pass' and not first['contract_evidence_satisfied']
    assert contract_gate(.12,'test',evidence_kind='digitized_measurement',contract_path=path)['status'] == 'insufficient_evidence'
    row['threshold_max'] = .1
    path.write_text(json.dumps({'metrics':[row]}))
    second = contract_gate(.12,'test',evidence_kind='independent_measurement',contract_path=path)
    assert second['status'] == 'fail' and second['contract_sha256'] != first['contract_sha256']
    path.write_text(json.dumps({'metrics':[row,row]}))
    with pytest.raises(ValueError):
        contract_gate(.12,'test',evidence_kind='independent_measurement',contract_path=path)


def test_repeatability_limits_distinguish_sem_from_population_sd():
    from protomotions.robot_configs.human_model.metrics import repeatability_limits
    result = repeatability_limits(2., .6, .75)
    assert result['sem'] == pytest.approx(.3)
    assert result['sem_percent_of_mean'] == pytest.approx(15)
    assert result['mdc95'] == pytest.approx(1.96*np.sqrt(2)*.3)
    assert repeatability_limits(2., .6, 1.)['sem'] == 0
    for args in ((0,1,.5),(1,-1,.5),(1,1,1.1),(1,1,np.nan)):
        with pytest.raises(ValueError):
            repeatability_limits(*args)


def test_residual_decomposition_does_not_hide_absolute_offset():
    from protomotions.robot_configs.human_model.metrics import residual_decomposition
    ref = np.array([1., 3., 9.])
    result = residual_decomposition(ref+10, ref, scale=20)
    assert result['absolute']['nrmse'] == .5
    assert result['centered_nrmse'] == 0
    assert result['bias_fraction_of_mse'] == 1
    varying = residual_decomposition(ref+np.array([8., 10., 12.]), ref, scale=20)
    assert varying['mse_identity_error_nm2'] < 1e-12
    assert varying['centered_rmse_nm'] == pytest.approx(np.sqrt(8/3))
    assert residual_decomposition(ref, ref, scale=20)['bias_fraction_of_mse'] is None
    with pytest.raises(ValueError):
        residual_decomposition([1, 2], [1, 2], scale=20)


def test_step_metrics_known_exponential_and_incomplete_response():
    t = np.linspace(-.1, .5, 6001)
    y = np.where(t >= 0, 1 - np.exp(-np.maximum(t, 0)/.04), 0)
    r = step_response_metrics(t, y, initial=0, target=1, onset=0)
    assert r["transition_10_90_s"] == pytest.approx(.04*np.log(9), abs=2e-6)
    assert r["settling_s"] == pytest.approx(-.04*np.log(.02), abs=.0001)
    falling = step_response_metrics(t, 1-y, initial=1, target=0, onset=0)
    assert falling == r
    short = step_response_metrics(t[:1200], y[:1200], initial=0, target=1, onset=0)
    assert short["t90_s"] is None and short["settling_s"] is None


def test_observed_rise_distinguishes_missing_onset_from_measurable_interval():
    from protomotions.robot_configs.human_model.metrics import observed_transition_time
    t = np.linspace(5, 5.5, 5001)
    y = 1-np.exp(-(t-5)/.04)
    assert observed_transition_time(t,y,initial=0,target=1) == pytest.approx(.04*np.log(9),abs=1e-6)
    assert observed_transition_time(t,-y,initial=0,target=-1) == pytest.approx(.04*np.log(9),abs=1e-6)
    assert observed_transition_time(t,y*.5,initial=0,target=1) is None
    assert observed_transition_time(t,y+.2,initial=0,target=1) is None


def test_impedance_requires_excitation_and_recovers_known_multisine():
    t = np.linspace(0, 3, 3001)
    frequencies = np.array([3., 7., 13.])
    phase = t[:, None] * frequencies
    q = np.sin(phase).sum(1)*.01
    qd = (np.cos(phase)*frequencies).sum(1)*.01
    qdd = (-np.sin(phase)*frequencies**2).sum(1)*.01
    torque = .03*qdd + 2*qd + 80*q + .3
    r = impedance_fit(q, qd, qdd, torque)
    assert r["stiffness_nm_rad"] == pytest.approx(80)
    assert r["damping_nms_rad"] == pytest.approx(2)
    assert r["inertia_kg_m2"] == pytest.approx(.03)
    with pytest.raises(ValueError):
        impedance_fit(np.sin(t), np.cos(t), -np.sin(t), torque)


def test_trace_uses_time_weighting_and_separates_passive_from_active_caps():
    t = [0, 1, 3]
    q = np.zeros((3, 1)); qd = np.ones((3, 1))
    act = np.ones((3, 1)); elastic = np.full((3, 1), 4.)
    damping = -np.ones((3, 1)); applied = act + elastic + damping
    r = torque_trace_metrics(t, q=q, qd=qd, requested=[[0], [2], [0]], active=act,
        elastic=elastic, damping=damping, applied=applied, lower=[-1], upper=[1], negative_cap=[1], positive_cap=[1])
    assert r["saturation_fraction"] == [.5]
    assert r["positive_work_j"] == [12]
    assert r["cap_violation_max_nm"] == [0]
    assert r["torque_accounting_max_nm"] == [0]
    assert r["damping_energy_injection_j"] == [0]


def test_rom_classification_denominators_and_radian_conversion():
    r = rom_boundary_metrics([0, .1], [0, 0], [True, False, True], [False, True, True])
    assert r["rom_boundary_error_deg"] == pytest.approx(np.rad2deg(.1))
    assert r["rom_false_accept_fraction"] == 1
    assert r["rom_false_reject_fraction"] == .5
    r = rom_boundary_metrics([0], [0], [True], [True])
    assert r["rom_false_accept_fraction"] is None


def test_cop_flight_is_excluded_but_missing_stance_is_visible():
    force = [[0,0,0],[0,0,100],[0,0,100]]
    r = contact_metrics(force,force,[[np.nan,np.nan],[0,0],[np.nan,0]],
                        [[np.nan,np.nan],[.01,0],[0,0]],mass_kg=60)
    assert r["cop_rmse_m"] == .01 and r["cop_coverage_fraction"] == .5
    assert r["grf_rmse_bw"] == [0,0,0]


def test_closed_work_and_wrapped_frequency_phase():
    phase = np.linspace(0,2*np.pi,10001)
    q = .1*np.sin(phase)
    torque = 5*q + 2*.1*np.cos(phase)
    assert cycle_work(q,torque) == pytest.approx(2*.01*np.pi,rel=1e-6)
    with pytest.raises(ValueError):
        cycle_work([0,.1,.2],[0,1,2])
    r = frequency_response_metrics([np.exp(1j*np.deg2rad(-179))],[np.exp(1j*np.deg2rad(179))])
    assert r["frequency_phase_error_deg"] == pytest.approx(2)


def test_summary_preserves_failed_evidence_and_separates_plant_requirements(tmp_path):
    import csv
    import json
    from protomotions.robot_configs.human_model.metrics import contract_gate
    from protomotions.robot_configs.human_model.validation import summarize

    evaluation = tmp_path / 'evaluation' / 'screening'
    evaluation.mkdir(parents=True)
    historical = evaluation / 'historical.json'
    original = json.dumps({'biological_and_walking_gates': {
        'completion_drop_pp': gate(100, 0, evidence_kind='independent_measurement'),
        'isometric_torque_nrmse': gate(.4, .15, evidence_kind='independent_measurement'),
    }})
    historical.write_text(original)
    screening_gate = contract_gate(.12, 'isometric_torque_nrmse',
                                  evidence_kind='digitized_measurement',
                                  required_evidence='digitized_measurement')
    (evaluation / 'screening.json').write_text(json.dumps({'curve': screening_gate}))
    summarize(tmp_path)
    result = json.loads((tmp_path / 'deliverables' / 'result.json').read_text())
    assert historical.read_text() == original
    assert result['status'] == 'candidate_failed_gate'
    assert result['recorded_gate_count'] == 3
    requirements = result['evaluation_requirements']
    plant = requirements['joint_level_plant']
    integration = requirements['supplementary_integration']
    assert 'completion_drop_pp' not in plant
    assert integration['completion_drop_pp']['scope'] == 'fixed_pretrained_controller_integration'
    assert integration['grf_rmse_bw']['scope'] == 'foot_contact_and_kinematic_integration'
    assert plant['isometric_torque_nrmse']['threshold_basis'] == 'engineering_acceptance_target'
    assert plant['active_cap_excess_max_nm']['required_evidence'] == 'numerical'
    assert plant['capacity_curve_nrmse']['implementation']['status'] == 'partial'
    assert 'No independent endurance/recovery validation' in plant['capacity_curve_nrmse']['implementation']['scope']
    assert plant['stiffness_nrmse']['implementation']['status'] == 'partial'
    assert plant['torque_sum_max_nm']['implementation']['status'] == 'implemented'
    assert len(plant) + len(integration) == 43
    assert all(row['status'] == 'not_evaluated' for bucket in requirements.values() for row in bucket.values())
    assert ',fail,' in (tmp_path / 'deliverables' / 'metrics_summary.csv').read_text()
    with (tmp_path / 'deliverables' / 'metrics_summary.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    screening = next(row for row in rows if row['metric_path'] == 'curve')
    assert screening['status'] == 'pass'
    assert screening['contract_evidence_satisfied'] == 'False'
    assert screening['contract_required_evidence'] == 'independent_measurement'
    assert screening['assay_required_evidence'] == 'digitized_measurement'
    assert screening['contract_sha256'] == screening_gate['contract_sha256']
    assert screening['threshold_basis'] == 'engineering_acceptance_target'
    # Old observations have no contract provenance; do not fabricate it.
    old = next(row for row in rows if row['metric_path'].endswith('completion_drop_pp'))
    assert old['contract_evidence_satisfied'] == ''
    assert old['contract_sha256'] == ''


def test_multiaxis_impedance_identifies_cross_terms_and_rejects_unexcited_axes():
    from protomotions.robot_configs.human_model.metrics import impedance_fit
    time = np.linspace(0,3,1000)
    frequencies = np.array([[1.3,2.1,3.4],[1.7,2.8,4.2]])
    phase = time[:,None,None]*frequencies[None,:,:]*2*np.pi
    omega = frequencies[None,:,:]*2*np.pi
    q = .01*np.sin(phase).sum(-1)
    qd = .01*(omega*np.cos(phase)).sum(-1)
    qdd = -.01*(omega**2*np.sin(phase)).sum(-1)
    inertia = np.array([[.12,.02],[.02,.07]])
    damping = np.array([[.9,.3],[.1,.6]])
    stiffness = np.array([[18.,7.],[4.,12.]])
    offset = np.array([.7,-.4])
    torque = qdd@inertia.T + qd@damping.T + q@stiffness.T + offset
    result = impedance_fit(q,qd,qdd,torque)
    for key,expected in [('inertia_kg_m2',inertia),('damping_nms_rad',damping),
                         ('stiffness_nm_rad',stiffness),('offset_nm',offset)]:
        np.testing.assert_allclose(result[key],expected,atol=1e-11)
    assert max(result['residual_rmse_nm']) < 1e-12
    assert result['n_dof'] == 2
    # Moving both axes identically cannot identify their separate cross terms.
    with pytest.raises(ValueError,match='identifiable'):
        impedance_fit(np.repeat(q[:,:1],2,axis=1),np.repeat(qd[:,:1],2,axis=1),
                      np.repeat(qdd[:,:1],2,axis=1),torque)
    with pytest.raises(ValueError,match='insufficient'):
        impedance_fit(q[:6],qd[:6],qdd[:6],torque[:6])


def test_fixed_inertia_identifies_stiffness_when_joint_estimation_is_ambiguous():
    t=np.linspace(0,3,1000);w=2*np.pi*2
    q=.01*np.sin(w*t);qd=.01*w*np.cos(w*t);qdd=-w*w*q
    torque=.12*qdd+.8*qd+17*q+.4
    with pytest.raises(ValueError,match='identifiable'):
        impedance_fit(q,qd,qdd,torque)
    result=impedance_fit(q,qd,qdd,torque,fixed_inertia=.12)
    assert result['inertia_mode']=='fixed'
    assert result['inertia_kg_m2']==.12
    assert result['stiffness_nm_rad']==pytest.approx(17)
    assert result['damping_nms_rad']==pytest.approx(.8)
    assert result['residual_rmse_nm']<1e-12
    # A fixed inertia is an external assumption, not identified by a good fit.
    wrong=impedance_fit(q,qd,qdd,torque,fixed_inertia=.13)
    assert wrong['stiffness_nm_rad']==pytest.approx(17+.01*w*w)
    assert wrong['residual_rmse_nm']<1e-12
    q2=np.column_stack([q,.01*np.sin(1.7*w*t)])
    v2=np.column_stack([qd,.017*w*np.cos(1.7*w*t)])
    a2=np.column_stack([qdd,-(1.7*w)**2*q2[:,1]])
    inertia=np.array([[.12,.02],[.02,.08]])
    stiffness=np.array([[17,3],[5,12]])
    torque2=a2@inertia.T+.8*v2+q2@stiffness.T
    result=impedance_fit(q2,v2,a2,torque2,fixed_inertia=inertia)
    np.testing.assert_allclose(result['stiffness_nm_rad'],stiffness,atol=1e-11)
    for invalid in (-.1,[.12],float('nan')):
        with pytest.raises(ValueError):
            impedance_fit(q,qd,qdd,torque,fixed_inertia=invalid)
    for invalid in (.12,[[1,2],[0,1]],[[1,2],[2,1]]):
        with pytest.raises(ValueError):
            impedance_fit(q2,v2,a2,torque2,fixed_inertia=invalid)


def test_shared_state_bound_detects_incompatible_curves_without_interpolation():
    from protomotions.robot_configs.human_model.metrics import shared_state_error_bound
    r = shared_state_error_bound([[0],[1]],[0,0],[[0],[1]],[40,40],scale=100)
    assert r['minimum_possible_max_curve_nrmse'] == pytest.approx(.2)
    # A midpoint model attains this bound; the denominator uses whole curves.
    assert curve_metrics([20,20],[0,0],scale=100)['nrmse'] == pytest.approx(.2)
    r = shared_state_error_bound([[0],[1]],[0,0],[[0],[2]],[40,40],scale=100)
    assert r['shared_samples'] == 1
    assert r['minimum_possible_max_curve_nrmse'] == pytest.approx(40/(200*np.sqrt(2)))
    r = shared_state_error_bound([[0]],[0],[[1e-12]],[40],scale=100)
    assert r['shared_samples'] == 0
    with pytest.raises(ValueError,match='duplicate'):
        shared_state_error_bound([[0],[0]],[0,1],[[0]],[40],scale=100)


def test_single_anchor_distinguishes_amplitude_from_shape_without_test_leakage():
    from protomotions.robot_configs.human_model.metrics import anchored_curve_metrics
    r=anchored_curve_metrics([10,20,30],[20,40,60],anchor_index=1,scale=100)
    assert r['amplitude_factor']==2 and r['evaluation_indices']==[0,2]
    assert r['anchored']['nrmse']==0
    changed=anchored_curve_metrics([10,20,30],[100,40,60],anchor_index=1,scale=100)
    assert changed['amplitude_factor']==r['amplitude_factor']
    assert changed['anchored']['nrmse']>0
    negative=anchored_curve_metrics([-10,-20,-30],[-20,-40,-60],anchor_index=1,scale=100)
    assert negative['anchored']['nrmse']==0
    with pytest.raises(ValueError):
        anchored_curve_metrics([0,20,30],[20,40,60],anchor_index=0,scale=100)


def test_passive_selection_preserves_tradeoffs_and_rejects_changed_support():
    from protomotions.robot_configs.human_model.validation import passive_candidate_selection
    def run(mean,errors):
        return {'candidate_study_macro_nrmse':mean,'folds':[{'held_out_study':'a','curves':[
            {'id':str(i),'candidate':{'n':5,'nrmse':e}} for i,e in enumerate(errors)]}]}
    a,b,c=run(.1,[.05,.4]),run(.12,[.12,.2]),run(.2,[.15,.5])
    result=passive_candidate_selection([('mean_best',a),('worst_best',b),('dominated',c)])
    states={r['source']:r['selection_status'] for r in result['candidates']}
    assert states=={'mean_best':'candidate','worst_best':'candidate','dominated':'archived'}
    assert result['runtime_adoption'] is False
    b['folds'][0]['curves'][0]['candidate']['n']=4
    assert passive_candidate_selection([('a',a),('b',b)])['status']=='incomparable'


def test_gentle_chapters_have_smooth_bounded_targets():
    from protomotions.robot_configs.human_model.validation import joint_demo_plan,joint_demo_target
    from protomotions.robot_configs.human_model.profile import load_profile
    profile=load_profile()
    for scenario,count in [('lower_body',4),('upper_body',3)]:
        chapters=joint_demo_plan(scenario)
        assert len(chapters)==count
        for chapter in chapters:
            trajectory=np.array([joint_demo_target(chapter,t) for t in np.linspace(0,8,1601)])
            lo,hi=np.deg2rad(profile['joints'][chapter['joint']]['rom_deg'])
            assert trajectory[:,0].min()>=lo and trajectory[:,0].max()<=hi
            assert abs(trajectory[:,1]).max()<np.deg2rad(60)
            assert np.allclose(trajectory[[0,-1]],0)
            for t in [2.5,5.5]:
                assert np.allclose(joint_demo_target(chapter,t)[1:],0)


def test_population_summary_matches_raw_people_and_rejects_repeat_sessions():
    from protomotions.robot_configs.human_model.metrics import pool_population_summaries, population_band_metrics
    cohorts=[np.array([[1.,4.],[2.,6.],[6.,8.]]),np.array([[4.,2.],[8.,4.]])]
    result=pool_population_summaries([len(c) for c in cohorts],
        [c.mean(axis=0) for c in cohorts],[c.std(axis=0,ddof=1) for c in cohorts],
        participant_groups=['study_a','study_b'])
    people=np.concatenate(cohorts)
    assert result['n']==5
    assert np.allclose(result['mean'],people.mean(axis=0))
    assert np.allclose(result['sd'],people.std(axis=0,ddof=1))
    with pytest.raises(ValueError,match='repeated people'):
        pool_population_summaries([3,3],[[2],[2]],[[1],[1]],participant_groups=['same','same'])
    band=population_band_metrics([10,12,16],[10,10,10],[2,2,2])
    assert band['within_1sd_fraction']==pytest.approx(2/3)
    assert band['within_2sd_fraction']==pytest.approx(2/3)
    assert band['rms_z']==pytest.approx(np.sqrt(10/3))
    with pytest.raises(ValueError):population_band_metrics([1],[1],[0])


def test_dynamometer_sweep_obeys_declared_slow_speed_and_smooth_stops():
    from protomotions.robot_configs.human_model.validation import population_fixture_target
    c=dict(start_deg=90,end_deg=10,speed_deg_s=5,duration_s=21)
    samples=np.array([population_fixture_target(c,t) for t in np.linspace(0,21,2101)])
    assert np.rad2deg(samples[:,0]).min()==pytest.approx(10)
    assert np.rad2deg(samples[:,0]).max()==pytest.approx(90)
    assert np.rad2deg(abs(samples[:,1])).max()==pytest.approx(5)
    for t in [0,2,19,21]:assert np.allclose(population_fixture_target(c,t)[1:],0)
    h=1e-5
    for t in [2.3,5.,18.5]:
        before=np.array(population_fixture_target(c,t-h));after=np.array(population_fixture_target(c,t+h))
        assert np.allclose((after[:2]-before[:2])/(2*h),population_fixture_target(c,t)[1:],atol=1e-8)


def test_population_assay_uses_corrected_effort_and_tare_not_command_buffer():
    from protomotions.robot_configs.human_model.metrics import population_assay_metrics
    time=np.arange(1,8,dtype=float);active=np.array([0,0,5,10,10,10,10.])[:,None]
    passive=3.;gravity=np.full_like(active,4.);fixture=gravity-active-passive
    trace=dict(time=time,dof_names=np.array(['joint']),chapter_index=np.zeros(7),
        q=np.zeros_like(active),qd=np.zeros_like(active),applied=active+passive,fixture=fixture,
        gravity=gravity,coriolis=np.zeros_like(active),inertia_accel=np.zeros_like(active))
    chapter=dict(label='test',joint='joint',kind='isometric',direction=1,start_s=0,
        duration_s=7,population_reference=dict(mean=[10],sd=[2]))
    result=population_assay_metrics(trace,[chapter])[0]
    assert result['predicted']==[10]
    assert result['torque_closure_max_nm']==0
    trace['fixture']=fixture-6
    # A constant measurement offset cancels under the tare but must remain
    # visible in the independent native torque closure diagnostic.
    assert population_assay_metrics(trace,[chapter])[0]['torque_closure_max_nm']==6


def test_small_harmonic_rig_has_continuous_derivatives_and_recovers_damping():
    from protomotions.robot_configs.human_model.validation import population_fixture_target
    from protomotions.robot_configs.human_model.metrics import population_assay_metrics
    c=dict(label='wrist',joint='wrist',kind='harmonic',start_deg=0,end_deg=0,
        duration_s=8,start_s=0,frequency_hz=3,amplitude_deg=10,
        population_reference=dict(mean=[.28],sd=[.07]))
    for t in [0,2,6,8]:assert np.allclose(population_fixture_target(c,t),0)
    for t in [2.3,3.,4.2,5.,5.7]:
        h=1e-6;before=np.array(population_fixture_target(c,t-h));after=np.array(population_fixture_target(c,t+h))
        assert np.allclose((after[:2]-before[:2])/(2*h),population_fixture_target(c,t)[1:],atol=1e-5)
    t=np.arange(1,241)/30;values=np.array([population_fixture_target(c,x) for x in t]);q=values[:,0,None];qd=values[:,1,None]
    effort=-1.5*q-.28*qd;zeros=np.zeros_like(q)
    trace=dict(time=t,q=q,qd=qd,dof_names=np.array(['wrist']),chapter_index=np.zeros(len(t)),
        applied=effort,fixture=-effort,gravity=zeros,coriolis=zeros,inertia_accel=zeros)
    r=population_assay_metrics(trace,[c])[0]
    assert r['predicted'][0]==pytest.approx(.28)
    assert r['identified_stiffness_nm_rad']==pytest.approx(1.5)
    trace['q']=np.roll(q,-1,axis=0);trace['qd']=np.roll(qd,-1,axis=0)
    assert population_assay_metrics(trace,[c],pre_step={'q':q,'qd':qd})[0]['predicted'][0]==pytest.approx(.28)
    assert abs(population_assay_metrics(trace,[c])[0]['predicted'][0]-.28)>.01


def test_physics_plateau_ripple_detects_motion_that_video_rate_aliases():
    from protomotions.robot_configs.human_model.metrics import plateau_stability_metrics
    time=1+np.arange(480)/480
    q=(.1*np.cos(2*np.pi*30*time))[:,None]
    qd=(-.1*2*np.pi*30*np.sin(2*np.pi*30*time))[:,None]
    assert np.ptp(q[::16])<1e-10
    c=dict(label='hold',joint='joint',kind='isometric',start_s=0,duration_s=2)
    result=plateau_stability_metrics(time,q,qd,[c],['joint'])[0]
    assert result['focus_peak_to_peak_deg']==pytest.approx(np.rad2deg(.2))
    assert result['samples']==480
def test_active_rom_drive_removes_external_assistance_before_human_cap():
    import torch
    from protomotions.robot_configs.human_model.validation import active_rom_drive
    q=torch.zeros((1,3));mass=torch.diag_embed(torch.tensor([[1.,2.,3.]]))
    gravity=torch.tensor([[10.,20.,30.]]);rig=torch.tensor([[100.,200.,300.]])
    requested,external=active_rom_drive({'active_kp_nm_rad':1000},1,q,q,1.,0.,0.,mass,gravity,rig)
    assert requested.tolist()==[[0.,1020.,0.]]
    assert external.tolist()==[[100.,0.,300.]]
    assert rig.tolist()==[[100.,200.,300.]]
    # Gravity remains inside the requested human effort, including at zero error.
    hold,_=active_rom_drive({'active_kp_nm_rad':1000},1,q,q,0.,0.,0.,mass,gravity,rig)
    assert hold[0,1]==20
    with pytest.raises(ValueError):
        active_rom_drive({'active_kp_nm_rad':float('nan')},1,q,q,0.,0.,0.,mass,gravity,rig)


def test_active_rom_metrics_detects_subframe_assistance_and_missing_audit():
    from protomotions.robot_configs.human_model.metrics import active_rom_metrics
    n=3840;t=np.arange(n)/480;ft=np.arange(1,241)/30
    q=np.full((n,2),np.pi/2);qd=np.zeros_like(q);torque=np.zeros_like(q)
    trace=dict(dof_names=['R_Knee_y','R_Hip_y'],body_names=['R_Knee','R_Ankle','R_Toe'],
        time=ft,chapter_index=np.zeros(240),q=q[::16],target=q[::16],contact_body_forces_w=np.zeros((240,3,3)),lower=np.zeros(2),upper=np.full(2,np.pi))
    sub=dict(time=t,q=q,qd=qd,active=torque,requested=torque,applied=torque,elastic=torque,damping=torque,
        negative_cap=np.full_like(q,1000),positive_cap=np.full_like(q,1000))
    audit=dict(time=t.copy(),chapter_index=np.zeros(n),focus=np.zeros(n),
        target_rad=np.full(n,np.pi/2),fixture_focus_nm=np.zeros(n))
    chapters=[dict(joint='R_Knee_y',label='knee',start_s=0,duration_s=8,start_deg=90,end_deg=90)]
    audit['fixture_focus_nm'][17]=.1
    assert active_rom_metrics(trace,sub,audit,chapters)[0]['test_axis_fixture_max_nm']==.1
    audit['time']=t[::16]
    with pytest.raises(ValueError,match='every physical substep'):
        active_rom_metrics(trace,sub,audit,chapters)


def test_suspended_gait_targets_are_smooth_bounded_and_alternating():
    from protomotions.robot_configs.human_model.validation import suspended_gait_target
    from protomotions.robot_configs.human_model.profile import load_profile
    joints=load_profile()['joints'];names=list(joints)
    q=np.array([suspended_gait_target(names,t)[0] for t in np.linspace(0,12,721)])
    bounds=np.deg2rad([joints[n]['rom_deg'] for n in names])
    assert (q>=bounds[:,0]).all() and (q<=bounds[:,1]).all()
    for t in [0.,12.]:
        _,v,a=suspended_gait_target(names,t)
        assert np.max(abs(v))==0 and np.max(abs(a))==0
    h=1e-5
    for t in [.7,2.,3.4,10.,11.2]:
        _,v,a=suspended_gait_target(names,t)
        qm,vm,_=suspended_gait_target(names,t-h)
        qp,vp,_=suspended_gait_target(names,t+h)
        np.testing.assert_allclose((qp-qm)/(2*h),v,atol=1e-7)
        np.testing.assert_allclose((vp-vm)/(2*h),a,atol=1e-4)
    for t in np.linspace(2,10,37):
        q,_,_=suspended_gait_target(names,t)
        assert np.rad2deg(q[names.index('L_Hip_y')]+q[names.index('R_Hip_y')])==pytest.approx(-10)
