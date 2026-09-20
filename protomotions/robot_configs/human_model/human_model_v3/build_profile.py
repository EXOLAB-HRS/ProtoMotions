"""Reproducible v3 candidate, explicit evidence and sensitivity scenarios."""
import copy
import json
from pathlib import Path

ROOT=Path(__file__).parent
TARGET_MASS_KG=63.31
TARGET_HEIGHT_M=1.754

# Reduced three-joint allocation derived at neutral from MyoSuite myotorso
# muscle moment arms and active forces.  The local capacities are grouped as
# lower=(L5/S1,L4/L5), middle=(L3/L4,L2/L3), upper=(L1/L2), then direction-wise
# scaled so the coupling-weighted three-joint moment reproduces Pan2025's
# measured whole-trunk MVC for the target mass.  These are coordinate-actuator
# caps; they do not claim to reproduce shared-muscle co-contraction.
TORSO_MUSCLE_ALLOCATION={
    'x': {
        'weights':[0.633467417538214,0.26570126039152586,0.10083132207026012],
        'negative_nm':[64.23676913727822,49.33664435121199,50.357280374310065],
        'positive_nm':[64.21513361920097,49.40046315810913,50.3250350203962],
    },
    'y': {
        'weights':[0.632,0.232,0.136],
        'negative_nm':[108.37707695443173,99.04302222892878,111.80188388005651],
        'positive_nm':[75.97606899159447,67.52364861379654,43.813514109643414],
    },
    'z': {
        'weights':[0.9138927049209201,0.06066610306808976,0.025441192010990172],
        'negative_nm':[42.868929283075474,41.624407887863114,40.54281894133084],
        'positive_nm':[42.84224822374433,41.870179462063234,40.91519228871841],
    },
}
TORSO_SEGMENT_INDEX={'Torso':0,'Spine':1,'Chest':2}

# A reported "MVC" is not automatically an active-muscle torque.  Keep the
# source measurement convention separate from the runtime active/passive split.
# Net or unspecified isometric measurements are matched at their documented
# test pose; they must not be used as a global clamp on active+passive torque.
MEASUREMENT_CONVENTIONS={
    'ANDERSON2007_DYNAMIC': {
        'source_quantity':'active_mvc_after_passive_and_gravity_subtraction',
        'runtime_mapping':'active_cap_direct',
        'test_pose_mapping':'angle_velocity_surface',
        'passive_correction':'none_source_already_separated',
    },
    'PAN2025_V3': {
        'source_quantity':'net_isometric_mvc_passive_baseline_not_reported',
        'runtime_mapping':'source_net_minus_model_passive_at_matched_test_pose',
        'test_pose_mapping':'model_neutral_q0_qd0_proxy_for_seated_neutral',
        'passive_correction':'zero_in_current_model_at_test_pose',
    },
    'VASAVADA2001_V3': {
        'source_quantity':'net_isometric_mvc_passive_baseline_not_reported',
        'runtime_mapping':'source_net_minus_model_passive_at_matched_test_pose',
        'test_pose_mapping':'model_neutral_q0_qd0_proxy_for_neutral_neck',
        'passive_correction':'zero_in_current_model_at_test_pose',
    },
    'MORIN2023': {
        'source_quantity':'max_isometric_muscle_torque_passive_baseline_not_reported',
        'runtime_mapping':'source_net_minus_model_passive_at_matched_test_pose',
        'test_pose_mapping':'model_q0_qd0_proxy; source_protocol_pose_not_an_anatomical_coordinate_fit',
        'passive_correction':'zero_in_current_model_at_test_pose',
    },
    'DANNESKIOLD2009': {
        'source_quantity':'max_isometric_torque_passive_baseline_not_reported',
        'runtime_mapping':'source_net_minus_model_passive_at_matched_test_pose',
        'test_pose_mapping':'model_q0_qd0_proxy; source_protocol_pose_transfer_unresolved',
        'passive_correction':'zero_in_current_model_at_test_pose',
    },
    'KOYKKA2025': {
        'source_quantity':'max_isometric_torque_passive_baseline_not_reported',
        'runtime_mapping':'source_net_minus_model_passive_at_matched_test_pose',
        'test_pose_mapping':'model_neutral_q0_qd0_proxy',
        'passive_correction':'zero_in_current_model_at_test_pose',
    },
    'DELP1996_V3': {
        'source_quantity':'max_isometric_torque_passive_baseline_not_reported',
        'runtime_mapping':'source_net_minus_model_passive_at_matched_test_pose',
        'test_pose_mapping':'model_neutral_q0_qd0_proxy',
        'passive_correction':'zero_in_current_model_at_test_pose',
    },
    'PROXY': {
        'source_quantity':'borrowed_direction_or_coordinate_proxy',
        'runtime_mapping':'engineering_active_cap_not_mvc_accounting',
        'test_pose_mapping':'unresolved',
        'passive_correction':'not_applicable',
    },
    'ASSUMPTION': {
        'source_quantity':'engineering_prior_or_structural_zero',
        'runtime_mapping':'engineering_active_cap_not_mvc_accounting',
        'test_pose_mapping':'not_applicable',
        'passive_correction':'not_applicable',
    },
}


def build():
    base=json.loads((ROOT.parent/'human_model_v2/profiles/healthy_adult_v2.json').read_text())
    trunk=json.loads((ROOT.parent/'human_model_v2/profiles/trunk_candidate.json').read_text())
    profile=copy.deepcopy(base)
    profile['id']='human_model_v3'
    profile['strength_reference_size']=[TARGET_MASS_KG,TARGET_HEIGHT_M]
    profile['strength_basis']={
        'target_mass_kg':TARGET_MASS_KG,'target_height_m':TARGET_HEIGHT_M,
        'population':'Healthy adult mixed-sex reference, not a measured individual or normative percentile',
        'selection_order':['source MVC normalization','documented anthropometric approximation','explicit unresolved proxy'],
        'gait_data_role':'Independent demand validation, NEVER a multiplier used to define MVC',
        'automatic_cap_increase':False,
        'mvc_accounting':'Active and passive are matched to the source quantity at the source test condition. Net MVC is not a global clamp on runtime total torque.',
    }
    profile['description']='Candidate on v2 kinematics: evidence-audited strength, explicit uncertainty, independently fixed PD. Not a certified population model.'
    profile['sources'].update({
        'PAN2025_V3':{'url':'https://doi.org/10.1186/s40001-025-02742-w','location':'Results: weight-normalized isometric medians, sex-average multiplied by 63.31kg; bilateral direction averages','scope':'Seated aggregate trunk MVC used only for absolute calibration'},
        'MYOSUITE_TORSO_DISTRIBUTION':{'url':'https://github.com/MyoHub/myosuite','location':'myotorso neutral-pose active muscle moments; source commit 93b0ca8f4ec90c9899ee7f05fee561e9911da91b','scope':'Relative lower/middle/upper serial allocation; not an independent human MVC'},
        'VASAVADA2001_V3':{'url':'https://pubmed.ncbi.nlm.nih.gov/11568704/','location':'C7 male/female mean: extension52/21 flexion30/15 lateral36/16 axial15/6 Nm','scope':'Equal-sex average, identical serial moment proxy; no posture curve validated'},
        'DELP1996_V3':{'url':'https://pubmed.ncbi.nlm.nih.gov/8884484/','location':'Peak radial11.0, ulnar9.5 Nm; task-specific isometric sample','scope':'Clinical direction mapped explicitly'},
    })
    audit={}
    for name,row in profile['joints'].items():
        state='measured_magnitude_coordinate_transfer'
        reason='Retain existing cited direction-specific measurement; no unvalidated OpenSim curve transfer.'
        if name.startswith(('Torso_','Spine_','Chest_')):
            axis=name[-1]
            segment=name.split('_')[0]
            index=TORSO_SEGMENT_INDEX[segment]
            allocation=TORSO_MUSCLE_ALLOCATION[axis]
            values=[allocation['negative_nm'][index],allocation['positive_nm'][index]]
            row['active_nm']=values
            row['note']='Pan2025 whole-trunk MVC calibrated; MyoSuite muscle-derived neutral serial allocation. Constant angle/speed envelope pending validation.'
            row['evidence']['active']='PAN2025_V3'
            row['evidence']['distribution']='MYOSUITE_TORSO_DISTRIBUTION'
            reason='Preserve measured aggregate MVC while using muscle-derived lower/middle/upper relative capacity instead of copying the whole-trunk value to every joint.'
            state='measured_aggregate_muscle_informed_serial_allocation_candidate'
        elif name.startswith(('Neck_','Head_')):
            row['active_nm']={'x':[26.,26.],'y':[36.5,22.5],'z':[10.5,10.5]}[name[-1]]
            row['evidence']['active']='VASAVADA2001_V3'
            row['note']='Equal-sex C7 moment; identical serial-coordinate proxy, not independent anatomical segment capacity.'
        elif name.endswith('Wrist_z'):
            row['active_nm']=[9.5,11.] if name.startswith('R') else [11.,9.5]
            row['evidence']['active']='DELP1996_V3'
            row['note']='Right positive radial, left negative radial. Constant peak envelope; posture dependence unvalidated.'
        elif '_Thorax_' in name:
            state='engineering_proxy_unresolved'
            reason='Retain20/20Nm existing shoulder-girdle actuator. Zeroing it is not justified by lack of data; no glenohumeral torque duplication assumption is accepted.'
        elif '_Toe_' in name:
            state='engineering_proxy_unresolved'
            reason='Retain5/10Nm lumped-toe prior. First-MTP2.1Nm cannot be transferred to the whole distal-foot axis without mapping all toes.'
        elif '_Subtalar_' in name:
            state='measured_proxy_unresolved'
            reason='Retain20.6/20.6Nm ankle inversion/eversion proxy; external-moment sign and weight-bearing protocol must be reconciled before adopting Ottaviani.'
        elif max(row['active_nm'])==0:
            state='structural_zero'
            reason='Near-lock elbow auxiliary axis or rigid hand axis; not missing physiological strength.'
        elif name.endswith(('Hip_y','Knee_y','Ankle_y')):
            state='measured_angle_velocity_with_boundary_hold'
            reason='Anderson young equal-sex source curves preserved; dynamic runtime envelope overrides static metadata. Outside-domain flags audited separately.'
        elif name.endswith('Shoulder_z'):
            state='measured_flexion_extension_proxy'
            reason='Measured70.1Nm flexion; same extension remains a proxy. Elevation-plane torque is not a validated flexion coordinate transform.'
        elif row['evidence']['active']=='PROXY':
            state='measured_proxy_coordinate_transfer'
            reason=row['note']
        source_values=list(row['active_nm'])
        norm={'method':'unscaled_source_missing_transfer_model','scale_negative_positive':[1.,1.],
              'source_mass_kg':None,'source_height_m':None,
              'uncertainty':'Unresolved anthropometric or coordinate transfer; not certified for target size'}
        if name.endswith(('Hip_y','Knee_y','Ankle_y')):
            norm.update(method='source_body_weight_times_height',
                        source_mass_kg=[72.8,62.1],source_height_m=[1.748,1.606],
                        scale_negative_positive=None,
                        uncertainty='Each cohort C1 is dimensionalized at target m*g*h BEFORE equal-sex averaging; metadata caps are not dynamic caps')
        elif name.startswith(('Torso_','Spine_','Chest_')):
            axis=name[-1];segment=name.split('_')[0];index=TORSO_SEGMENT_INDEX[segment]
            aggregate={'x':[.93*TARGET_MASS_KG]*2,'y':[1.685*TARGET_MASS_KG,1.10*TARGET_MASS_KG],'z':[.675*TARGET_MASS_KG]*2}[axis]
            norm.update(method='source_nm_per_kg_with_muscle_informed_serial_allocation',
                        uncertainty='MyoSuite neutral muscle geometry supplies relative allocation; shared-muscle activation and posture dependence remain unresolved')
            norm['source_aggregate_negative_positive_nm']=aggregate
            norm['source_negative_positive_nm_per_kg']=[v/TARGET_MASS_KG for v in aggregate]
            norm['serial_coordinate_weight']=TORSO_MUSCLE_ALLOCATION[axis]['weights'][index]
            norm['allocation_model']='MyoSuite myotorso neutral active moments, commit 93b0ca8f4ec90c9899ee7f05fee561e9911da91b'
        elif name.startswith(('Neck_','Head_')):
            male=TARGET_MASS_KG*TARGET_HEIGHT_M/(77.*1.77)
            female=TARGET_MASS_KG*TARGET_HEIGHT_M/(65.*1.64)
            data={'x':[(36,16),(36,16)],'y':[(52,21),(30,15)],'z':[(15,6),(15,6)]}[name[-1]]
            norm['source_direction_cohort_nm']=data
            row['active_nm']=[(m*male+f*female)/2 for m,f in data]
            norm.update(method='body_mass_times_height_proxy',source_mass_kg=[77.,65.],source_height_m=[1.77,1.64],
                        scale_negative_positive=[b/a for a,b in zip(source_values,row['active_nm'])],
                        uncertainty='Geometric m*h approximation, NOT the paper neck-geometry regression; no target neck circumference available')
        elif row['evidence']['active']=='MORIN2023' or name.endswith(('Shoulder_z','Subtalar_x')):
            scale=TARGET_MASS_KG/71.8
            row['active_nm']=[v*scale for v in source_values]
            norm.update(method='body_mass_proxy',source_mass_kg=71.8,scale_negative_positive=[scale,scale],
                        uncertainty='Aggregate-cohort Nm/kg approximation; age/sex/height and muscle-mass residuals unresolved')
        elif name.endswith(('Hip_x','Shoulder_x')):
            # Only abduction originates in Morin; adduction remains a separate source.
            index=1 if name.startswith('L_') else 0
            scale=TARGET_MASS_KG/71.8
            row['active_nm'][index]*=scale
            scales=[1.,1.];scales[index]=scale
            norm.update(method='direction_specific_mixed_source',source_mass_kg={'Morin_abduction':71.8,'Danneskiold_adduction':None},
                        scale_negative_positive=scales,
                        uncertainty='Morin abduction mass-scaled; Danneskiold adduction unscaled pending source unit/anthropometry resolution')
        elif max(source_values)==0:
            norm.update(method='structural_zero',uncertainty='No independent actuator by design')
        elif row['evidence']['active']=='ASSUMPTION':
            norm.update(method='engineering_prior_at_target_geometry',uncertainty='No measured MVC: target-model engineering prior only')
        norm['pre_normalization_negative_positive_nm']=source_values
        if row['active_nm']!=source_values:
            row['note']+=' MVC size transfer: '+norm['method']+'; source values retained in joint_decisions.'
        convention=copy.deepcopy(MEASUREMENT_CONVENTIONS.get(row['evidence']['active'],MEASUREMENT_CONVENTIONS['PROXY']))
        convention['model_test_q_rad']=0.0 if convention['passive_correction']=='zero_in_current_model_at_test_pose' else None
        convention['model_test_qd_rad_s']=0.0 if convention['passive_correction']=='zero_in_current_model_at_test_pose' else None
        convention['model_passive_test_nm']=0.0 if convention['passive_correction']=='zero_in_current_model_at_test_pose' else None
        convention['numeric_cap_change_nm']=0.0
        convention['scope_note']='Zero correction means the current passive law is zero at the mapped isometric test pose; it does not mean passive torque is zero during gait or near ROM limits.'
        audit[name]={'negative_peak_metadata_nm':row['active_nm'][0],
                     'positive_peak_metadata_nm':row['active_nm'][1],
                     'status':state,'decision':reason,'evidence':copy.deepcopy(row['evidence']),
                     'rom_deg':row['rom_deg'],'normalization':norm,
                     'measurement_convention':convention}
    pd={}
    for name,row in base['joints'].items():
        cap=max(row['active_nm'])
        # Structural-zero gains never create active torque, and are set to zero.
        kp=cap/1.5;kd=kp*.1
        if name.startswith(('Torso_','Spine_','Chest_')):kp*=4;kd*=.5
        pd[name]={'kp':kp,'kd':kd}
    scenarios={'pd_gains':pd}
    for label,scale in [('low',.8),('reference',1.),('high',1.2)]:
        parameters=copy.deepcopy(trunk['human_model_parameters'])
        parameters['active_strength_scale']={n:[scale,scale] for n,r in profile['joints'].items() if max(r['active_nm'])>0}
        scenarios[label]={'status':'sensitivity_scenario_not_population_percentile',
                          'scale':scale,'parameters':parameters}
    dest=ROOT/'profiles';dest.mkdir(exist_ok=True)
    for name,data in [('healthy_adult_v3.json',profile),('strength_scenarios.json',scenarios),('joint_decisions.json',audit)]:
        (dest/name).write_text(json.dumps(data,indent=2)+'\n')


if __name__=='__main__':build()
