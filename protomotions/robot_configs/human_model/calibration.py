# SPDX-License-Identifier: Apache-2.0
"""Import source-tagged public curves and fit conservative passive candidates.

This operates on numeric research data, never executes downloaded MATLAB code.
Silder curves are digitized fitted curves and cannot be independent confirmation
of a profile whose parameters already use that study.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from scipy.io import loadmat
from scipy.optimize import nnls, least_squares
import torch

from .dynamics import HumanJointModel
from .metrics import curve_metrics, contract_gate, shared_state_error_bound, anchored_curve_metrics, residual_decomposition, observed_transition_time, population_band_metrics
from .profile import load_profile


def capacity_interval_from_rounded_targets(targets, fractions, *, rounding_step):
    """Invert positive MVC-fraction targets; retain displayed-value quantization.

    Closed intervals are conservative at rounding ties. This is not measurement
    uncertainty, and inconsistent targets must not be averaged into a capacity.
    """
    targets = np.asarray(targets, dtype=float)
    fractions = np.asarray(fractions, dtype=float)
    if (targets.ndim != 1 or not targets.size or targets.shape != fractions.shape
            or not np.isfinite(targets).all() or not np.isfinite(fractions).all()
            or (targets <= 0).any() or (fractions <= 0).any() or (fractions > 1).any()
            or not math.isfinite(rounding_step) or rounding_step <= 0):
        raise ValueError('positive paired targets/fractions and rounding step required')
    lower = float(np.maximum(0, (targets-rounding_step/2)/fractions).max())
    upper = float(((targets+rounding_step/2)/fractions).min())
    if lower > upper:
        raise ValueError('rounded targets imply incompatible capacity intervals')
    return [lower, upper]


def screen_wind_rouse_strength(output):
    """Table2 target-derived MVC intervals at neutral ankle, knee flexed 90deg."""
    output = Path(output)
    source = output/'scratch/wind_rouse2020_supplement.pdf'
    target = output/'evaluation/screening/healthy_adult_v1_wind_rouse_strength_seed_none.json'
    if target.exists():
        raise FileExistsError(target)
    text = subprocess.run(['pdftotext','-layout',str(source),'-'],check=True,
                          capture_output=True,text=True).stdout
    blocks = re.split(r'Supplementary Table \d+:',text)
    rows = []
    for line in blocks[1].splitlines():
        tokens = line.split()
        if len(tokens) == 10 and tokens[0].isdigit():
            rows.append([int(tokens[0]), *map(float,tokens[1:])])
    if [row[0] for row in rows] != list(range(1,15)):
        raise ValueError('Expected the 14 subjects in source Table2')
    references = []
    for row in rows:
        if row[1] >= 0 or row[4] >= 0 or row[7] <= 0:
            raise ValueError('Source torque-direction convention changed')
        references.append({'source_subject':row[0], 'source_table_row':row[1:],
            'plantarflexion_nm':capacity_interval_from_rounded_targets([-row[1],-row[4]],[.2,.4],rounding_step=.1),
            'dorsiflexion_nm':capacity_interval_from_rounded_targets([row[7]],[.2],rounding_step=.1)})
    profile=load_profile(); names=list(profile['joints']); i=names.index('R_Ankle_y')
    q=torch.zeros((14,len(names)),dtype=torch.float64)
    q[:,names.index('R_Knee_y')]=math.pi/2
    params=json.loads(Path(__file__).with_name('candidate_parameters.json').read_text())['strength']
    variants={}
    for label,features in [('baseline',()),('strength',('strength',)),
                           ('strength_coupling',('strength','strength_coupling'))]:
        model=HumanJointModel(profile,names,dtype=torch.float64,features=features)
        negative,positive,_=model.strength_caps(q,torch.zeros_like(q)); variants[label]={}
        for direction,caps in [('plantarflexion',positive),('dorsiflexion',negative)]:
            bounds=np.array([r[direction+'_nm'] for r in references]); prediction=caps[:,i].numpy()
            # Minimum possible error to any unrounded MVC inside each interval.
            distance=np.maximum.reduce([bounds[:,0]-prediction,prediction-bounds[:,1],np.zeros(14)])
            coefficients=params['directions'][direction]['coefficients']
            scale=float(np.mean([c[0]*s['mass_kg']*s['height_m']*params['gravity_m_s2']
                                 for c,s in zip(coefficients,params['cohorts'])]))
            lower_bound=float(np.sqrt(np.mean(distance**2))/scale)
            variants[label][direction]={'predicted_capacity_nm':prediction.tolist(),
                'normalization_nm':scale,'distance_to_rounding_interval_nm':distance.tolist(),
                'minimum_possible_nrmse':lower_bound,
                'target_lower_bound_gate':contract_gate(lower_bound,'isometric_torque_nrmse',
                    evidence_kind='source_target_derived',required_evidence='source_target_derived')}
    result={'status':'candidate_failed_gate','selection_status':'candidate_failed_gate',
        'scope':'Unpersonalized strength screening against target-derived MVC intervals; no independent confirmation',
        'references':references,'variants':variants,
        'source_url':'https://www.nature.com/articles/s41598-020-67135-x',
        'source_sha256':{str(source.relative_to(output)):hashlib.sha256(source.read_bytes()).hexdigest()},
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'condition':{'knee_flexion_deg':90,'ankle_neutral':True,'side':'right','male_n':7,'female_n':7},
        'limitations':['Intervals cover table rounding only, not measurement uncertainty',
            'Group composition matches equal-sex weighting, but subject sex/size are not identified by row',
            'Target-tracking achieved means are not MVC and are not used to infer capacity',
            'Point capacity at one posture, not a torque-angle curve or coactivation-stiffness law'],
        'runtime_adoption':False}
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({label:{direction:r['minimum_possible_nrmse'] for direction,r in values.items()}
                      for label,values in variants.items()}))


def read_catalog(path, sheet=2):
    """Read values AND original style ids. No workbook writes or formula execution."""
    ns = {'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(path) as z:
        strings = [''.join(e.itertext()) for e in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        rows = []
        for row in ET.fromstring(z.read(f'xl/worksheets/sheet{sheet}.xml')).findall('.//m:sheetData/m:row',ns):
            values, styles = {}, {}
            for c in row.findall('m:c',ns):
                address = c.attrib['r']; col = re.sub(r'\d','',address)
                value = c.find('m:v',ns)
                value = value.text if value is not None else ''
                if c.attrib.get('t') == 's':
                    value = strings[int(value)]
                values[col] = value; styles[col] = c.attrib.get('s','0')
            rows.append({'values':values,'styles':styles,'excel_row':int(row.attrib['r'])})
        return rows


def _curves(value):
    value = np.asarray(value)
    if value.dtype == object:
        return [np.asarray(a,dtype=float) for a in value]
    return [np.asarray(value,dtype=float)]


def import_toe_population(source, output):
    """Source workbook -> within-person visit means -> between-person SD.

    No extrapolation. The 0.001 Nm slack anchors are fitted pseudo-observations
    and excluded. Summed hallux/lesser moments are explicitly a lumped-toe proxy:
    their physical axes differ and electrical twitch maxima are not MVC caps.
    """
    source,output=Path(source),Path(output)
    rows={r['excel_row']:r['values'] for r in read_catalog(source,sheet=4)}
    def col(i):
        text=''
        while i:
            i,r=divmod(i-1,26);text=chr(65+r)+text
        return text
    grid=np.arange(20.,61.,5.)
    people=[]; provenance=[]
    for person in range(9):
        visits=[]
        for visit in range(2):
            pair=[]
            a=2+person*4+visit*2;b=a+1
            for start,end in [(8,26),(34,51)]:
                values=[]
                for ri in range(start,end+1):
                    row=rows.get(ri,{})
                    try:values.append([float(row[col(a)]),float(row[col(b)])])
                    except (KeyError,ValueError):pass
                xy=np.array(values);xy=xy[np.argsort(xy[:,0])]
                unique=np.unique(xy[:,0])
                # Replicate observations at an identical angle stay within visit.
                torque=np.array([xy[xy[:,0]==x,1].mean() for x in unique])
                curve=np.interp(grid,unique,torque,left=np.nan,right=np.nan)
                pair.append(curve)
                provenance.append(dict(person=person+1,visit=visit+1,
                    cells=f'{col(a)}{start}:{col(b)}{end}',observations=len(xy),
                    observed_angle_range_deg=[float(unique[0]),float(unique[-1])]))
            visits.append(pair[0]+pair[1])
        # Both visits required, so the subject's session selection never varies.
        people.append(np.mean(visits,axis=0))
    people=np.array(people);counts=np.isfinite(people).sum(axis=0)
    means=[];sds=[]
    for j,n in enumerate(counts):
        valid=people[np.isfinite(people[:,j]),j]
        means.append(float(valid.mean()) if n>=2 else None)
        sds.append(float(valid.std(ddof=1)) if n>=2 else None)
    result=dict(status='candidate_proxy',source_url='https://doi.org/10.1242/jeb.249816',
        workbook=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        cohort=dict(n=9,mass_mean_sd_kg=[71,11],height_mean_sd_m=[1.74,.07],age_mean_sd_years=[28,5]),
        sheet='Passive T-A Data',angle_deg=grid.tolist(),n_per_angle=counts.tolist(),
        mean_nm=means,sd_nm=sds,spread='between-person sample SD after averaging two visits within person',
        individual_nm=[[None if not np.isfinite(v) else float(v) for v in row] for row in people],
        method='Piecewise linear interpolation inside observed support only; hallux + lesser digits per person/visit, then visit mean',
        limitations=['Lumped-toe projection proxy: anatomical hallux and lesser axes differ.',
                      'n=9 below preferred n>=10; per-angle n may be smaller.',
                      'Missing raw body specs prevent individual anthropometric filtering.',
                      'Fitted slack anchors rows 7 and 33 excluded; not observed ROM endpoints.'],provenance=provenance)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({'output':str(output),'n_per_angle':counts.tolist(),'mean_nm':means,'sd_nm':sds}))


def fit_population_knee(source, output):
    """Two-parameter conservative hamstrings candidate, never global adoption.

    Fit alternating angles of one declared baseline cohort; other angles test
    interpolation only. Preserve hip:knee coupling ratio and positive energy.
    No source SD, ROM, strength cap or target gate is adjusted by the optimizer.
    """
    source,output=Path(source),Path(output)
    record=json.loads(source.read_text());ref=record['passive']
    profile=load_profile();names=list(profile['joints']);idx=names.index(ref['joint'])
    q=torch.zeros((len(ref['mean_nm']),len(names)),dtype=torch.float64)
    q[:,names.index('R_Hip_y')]=-math.radians(ref['hip_flexion_deg'])
    q[:,idx]=torch.tensor(ref['clinical_flexion_deg'],dtype=torch.float64)*math.pi/180
    mean=np.array(ref['mean_nm']);sd=np.array(ref['sd_nm']);fit=np.arange(len(mean))%2==0
    base=HumanJointModel(profile,names,dtype=torch.float64)
    def candidate(x):
        p=copy.deepcopy(profile)
        for c in p['passive_exponentials']:
            if c['name'].endswith('_hamstrings'):
                c['coeff_per_rad']={n:v*math.exp(x[0]) for n,v in c['coeff_per_rad'].items()}
                c['energy_scale_j']*=math.exp(x[1])
                c['source']='KANEDA2020_CANDIDATE'
        p['sources']['KANEDA2020_CANDIDATE']={'url':record['source_url'],'cohort':'17 men, control pre; two-parameter population calibration, not independent validation','location':'Table 2 knee passive torque; coupled ratio retained from Silder basis'}
        return p
    def residual(x):
        m=HumanJointModel(candidate(x),names,dtype=torch.float64)
        return ((m.elastic_torque(q)[:,idx].numpy()-mean)/sd)[fit]
    opt=least_squares(residual,[math.log(.5),0.],bounds=([math.log(.1),-8],[math.log(2),8]))
    fitted=candidate(opt.x);model=HumanJointModel(fitted,names,dtype=torch.float64)
    measured=model.elastic_torque(q)[:,idx].numpy()
    report=dict(status='candidate',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        method='SD-weighted least squares on alternating angle indices; positive energy scale and common hamstrings exponent scaling only',
        fit_indices=np.flatnonzero(fit).tolist(),heldout_angle_indices=np.flatnonzero(~fit).tolist(),
        scope='Single-cohort calibration and held-out angle interpolation. No independent people; other hip/knee postures require regression.',
        exponent_scale=math.exp(opt.x[0]),energy_scale=math.exp(opt.x[1]),
        baseline=population_band_metrics(base.elastic_torque(q)[:,idx].numpy(),mean,sd),
        candidate=population_band_metrics(measured,mean,sd),
        heldout_angles=population_band_metrics(measured[~fit],mean[~fit],sd[~fit]),
        simulated_nm=measured.tolist(),profile_path=str(output/'configs/population_passive_candidate.json'))
    for path,value in [(output/'configs/population_passive_candidate.json',fitted),(output/'evaluation/screening/knee_population_fit.json',report)]:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
    print(json.dumps(report))


def import_curves(mat_path,catalog_path):
    rows = loadmat(mat_path,simplify_cells=True)['dataset_isometric_passive']
    catalog = read_catalog(catalog_path)[1:]
    if len(rows) != len(catalog):
        raise ValueError('MAT/catalog record counts differ')
    model = HumanJointModel(load_profile(),list(load_profile()['joints']),dtype=torch.float64)
    result = []
    mappings = {
        'Hip extension/flexion':('L_Hip_y',-1,'L_Knee_y',1),
        'Knee extension/flexion':('L_Knee_y',1,'L_Hip_y',-1),
        'Ankle plantar-/dorsiflexion':('L_Ankle_y',-1,'L_Knee_y',1),
        'Ankle plantarflexion':('L_Ankle_y',-1,'L_Knee_y',1),
        'Ankle dorsiflexion':('L_Ankle_y',-1,'L_Knee_y',1),
    }
    for number,(row,meta) in enumerate(zip(rows,catalog),1):
        # Known encoding losses in the published MAT; retain both originals.
        aliases = {'Ate? et al. (2018)':'Ateş et al. (2018)',
                   "O'Brien et al. (2009b)":'O’Brien et al. (2009b)',
                   'Oliveira & Gon?alves (2009)':'Oliveira & Gonçalves (2009)'}
        expected_number = str(number) if number <= 314 else f'S{number-314}'
        if meta['values']['B'] != aliases.get(row['reference'],row['reference']) or meta['values']['A'] != expected_number:
            raise ValueError('MAT/catalog row alignment failed')
        reasons = []
        mapping = mappings.get(row['measureObject'])
        if mapping is None: reasons.append('unsupported_anatomical_mapping')
        if row['measureType'] not in ('Active','Passive'): reasons.append('total_torque_cannot_substitute_for_active')
        if row['subjectType'] not in ('YM','YF','YB'): reasons.append('outside_young_nonathlete_cohort')
        for ci,values in enumerate(_curves(row['measureValue'])):
            reason = list(reasons)
            if values.ndim != 2 or values.shape[0] != 3:
                raise ValueError('expected [primary angle deg,secondary angle deg,torque Nm]')
            if values.shape[1] < 3: reason.append('fewer_than_three_points')
            if not np.isfinite(values).all(): reason.append('missing_angle_or_torque')
            rec = {'id':f'isometric_passive_{number:03d}_curve{ci}', 'dataset_number':number,'curve_index':ci,
                   'reference':row['reference'],'catalog_reference':meta['values']['B'],
                   'catalog_id':meta['values']['A'],'doi':meta['values'].get('M'), 'subject':row['subject'],
                   'subject_type':row['subjectType'],'measurement':row['measurement'],'measure_type':row['measureType'],
                   'object':row['measureObject'],'metadata':meta,
                   'evidence_kind':'digitized_fitted_curve' if 'fitted' in meta['values'].get('L','').lower() else 'digitized_measurement',
                   'reasons':reason,'status':'candidate' if not reason else 'archived'}
            if not reason:
                joint,sign,adj,adj_sign=mapping
                q=np.zeros((values.shape[1],len(model.names)))
                q[:,model.names.index(joint)]=np.deg2rad(values[0])*sign
                q[:,model.names.index(adj)]=np.deg2rad(values[1])*adj_sign
                if joint=='L_Knee_y':
                    tertiary=row['tertiaryDoF']
                    if isinstance(tertiary,np.ndarray): tertiary=str(tertiary[ci])
                    match=re.fullmatch(r'Ankle dorsiflexion\s+(-?[\d.]+)°',str(tertiary))
                    if match is None: reason.append('unresolved_ankle_angle_for_knee_coupling')
                    else: q[:,model.names.index('L_Ankle_y')]=-math.radians(float(match[1]))
                rec.update({'joint':joint,'q_rad':q.tolist(),'torque_nm':(values[2]*sign).tolist(),
                            'primary_angle_deg':values[0].tolist(),'secondary_angle_deg':values[1].tolist(),
                            'rom_covered_fraction':float(np.mean(((q>=model.lower.numpy()-1e-8)&(q<=model.upper.numpy()+1e-8)).all(1)))})
                if reason: rec['status']='archived'
            result.append(rec)
    return result


def isokinetic_joint_state(values, joint, torque_direction):
    """Map verified concentric-positive deg/s data, NOT the MAT header's sign.

    Anatomical torque signs must be supplied from the named movement, including
    at zero measured torque. Positive mechanical power denotes shortening.
    """
    angle_sign = {'L_Hip_y': -1, 'L_Knee_y': 1, 'L_Ankle_y': -1}[joint]
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] != 3 or not np.isfinite(values).all():
        raise ValueError('finite [concentric-positive deg/s, anatomical angle deg, torque Nm] required')
    if torque_direction not in (-1, 1) or (values[2]*torque_direction < 0).any():
        raise ValueError('torque disagrees with declared anatomical movement')
    return (np.deg2rad(values[1])*angle_sign,
            np.deg2rad(values[0])*torque_direction*angle_sign,
            values[2]*angle_sign)


def screen_isokinetic_velocity(output, source='holzer'):
    """Neutral-angle PF screening; source audit levels are retained explicitly."""
    specifications = {
        'holzer': (252, 'Holzer et al. (2023)', 'holzer_preloaded'),
        'barber': (247, 'Barber et al. (2013)', 'barber_concentric_eccentric')}
    number, reference_name, slug = specifications[source]
    output = Path(output)
    target = output/f'evaluation/screening/healthy_adult_v1_{slug}_velocity_seed_none.json'
    if target.exists():
        raise FileExistsError(target)
    mat = output/'scratch/chen2025_datasets.mat'
    catalog = output/'scratch/chen2025_catalog.xlsx'
    row = loadmat(mat, simplify_cells=True)['dataset_isokinetic'][number-1]
    meta = read_catalog(catalog, sheet=3)[number]
    if row['reference'] != reference_name or meta['values']['A'] != str(number) or meta['values']['B'] != row['reference']:
        raise ValueError('Source/catalog alignment changed')
    values = np.asarray(row['measureValue'], dtype=float)
    if row['measureObject'] != 'Ankle plantarflexion' or row['tertiaryDoF'] != 'Knee flexion 0°' or not (values[1] == 0).all():
        raise ValueError('Audited neutral-angle extended-knee protocol changed')
    angles, velocity, reference = isokinetic_joint_state(values, 'L_Ankle_y', -1)
    profile = load_profile(); names = list(profile['joints']); i = names.index('L_Ankle_y')
    q = torch.zeros((len(angles), len(names)), dtype=torch.float64); q[:,i] = torch.from_numpy(angles)
    qd = torch.zeros_like(q); qd[:,i] = torch.from_numpy(velocity)
    reference_model = HumanJointModel(profile, names, dtype=torch.float64, features=('strength',))
    # Same frozen Anderson calibration normalization as the isometric screens.
    params = json.loads(Path(__file__).with_name('candidate_parameters.json').read_text())['strength']
    scale = float(np.mean([c[0]*cohort['mass_kg']*cohort['height_m']*params['gravity_m_s2']
                           for c,cohort in zip(params['directions']['plantarflexion']['coefficients'], params['cohorts'])]))
    valid = ~reference_model.strength_caps(q, qd, directional_domain=True)[2][:,i,1]
    valid &= ((q >= reference_model.lower) & (q <= reference_model.upper)).all(-1)
    if valid.sum() < 3:
        raise ValueError('Insufficient source-domain samples')
    mask = valid.numpy()
    variants = {}
    options_by_name = [('baseline', {}), ('strength', {'features': ('strength',)})]
    if source == 'holzer':
        options_by_name.append(('cohort_size_matched', {'features': ('strength',), 'strength_cohort': 'male', 'strength_reference_size': (80, 1.8)}))
    for name, options in options_by_name:
        model = HumanJointModel(profile, names, dtype=torch.float64, **options)
        neg, pos, outside = model.strength_caps(q, qd)
        pred = pos[:,i].numpy()
        metrics = curve_metrics(pred[mask], reference[mask], scale=scale)
        anchors = np.flatnonzero(values[0,mask] == 0)
        if len(anchors) != 1:
            raise ValueError('Exactly one measured zero-speed reference required')
        variants[name] = {'predicted_nm': pred.tolist(), 'absolute': metrics,
                          'comparison_target_gate': contract_gate(metrics['nrmse'], 'isokinetic_torque_nrmse', evidence_kind='digitized_measurement', required_evidence='digitized_measurement'),
                          'bias_target_gate': contract_gate(abs(metrics['bias'])/scale, 'torque_bias_fraction', evidence_kind='digitized_measurement', required_evidence='digitized_measurement'),
                          'zero_speed_anchor': anchored_curve_metrics(pred[mask], reference[mask], anchor_index=int(anchors[0]), scale=scale)}
        variants[name]['contraction_conditions'] = {}
        for condition, condition_mask in (('concentric', values[0]>0), ('eccentric', values[0]<0)):
            selected = mask & condition_mask
            variants[name]['contraction_conditions'][condition] = {
                'indices': np.flatnonzero(selected).tolist(),
                'absolute': curve_metrics(pred[selected], reference[selected], scale=scale) if selected.any() else None}
    result = {'status': 'candidate_failed_gate', 'scope': 'Neutral-angle PF screening; no independent confirmation',
              'source': meta, 'source_mat_primary_label': row['primaryDoF'],
              'velocity_convention': 'Concentric positive per Chen2025 Methods; MAT primary label is reversed',
              'source_sha256': {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (mat,catalog)},
              'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'normalization_nm': scale, 'angle_rad': angles.tolist(), 'velocity_rad_s': velocity.tolist(),
              'valid_indices': np.flatnonzero(mask).tolist(), 'extrapolated_indices': np.flatnonzero(~mask).tolist(),
              'screening_domain': 'PF directional source domain only; all samples retained, extrapolation excluded from gate',
              'reference_nm': reference.tolist(), 'variants': variants,
              'protocol': {'preload': '95% of maximum fixed-end torque at 15 degrees dorsiflexion',
                           'torque_correction': 'Angle-specific passive and gravity correction confirmed in primary Methods',
                           'velocity_grid': 'Catalog preset speeds; actual camera-derived velocities not substituted',
                           'primary_url': 'https://www.nature.com/articles/s41598-023-33643-9'},
              'limitations': ['No tendon preload state in current plant', 'Preset versus actual velocity mismatch unresolved',
                              'Zero-speed anchoring is within-cohort calibration; not independent subject validation',
                              'Right-leg measurement mapped to symmetric left model'],
              'runtime_adoption': False}
    if source == 'barber':
        result['protocol'] = {
            'primary_url': 'https://pubmed.ncbi.nlm.nih.gov/23886750/',
            'primary_audit_level': 'Abstract confirms concentric/eccentric contractions; full Methods not obtained',
            'catalog_conditions': 'Neutral ankle, knee extended, passive/gravity corrected, EMG-monitored',
            'preload': 'unresolved', 'velocity_grid': 'Catalog speeds; actual versus prescribed unresolved'}
        result['limitations'] = [
            'Detailed Methods and figure mapping not verified independently of catalog',
            'Mixed 10M/8F group compared with default equal-sex model; no cohort-match claim',
            'Zero-speed anchoring uses this cohort and is not independent confirmation',
            'Preload and actual velocity conditions unresolved; no parameter adoption']
    target.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'path': str(target), 'metrics': {k: v['absolute'] for k,v in variants.items()}}))


def screen_holzer_velocity(output):
    """Compatibility entry point for the original source assay."""
    return screen_isokinetic_velocity(output, 'holzer')


def audit_holzer_measured_velocity(output):
    """Paired speed sensitivity using author-tabulated individual force ratios.

    Ratios are not absolute joint torques; conversion assumes the fixed-angle
    force/torque proportionality used by the source. No parameters are fitted.
    """
    output = Path(output)
    target = output/'evaluation/screening/healthy_adult_v1_holzer_measured_velocity_seed_none.json'
    if target.exists():
        raise FileExistsError(target)
    workbook = output/'scratch/holzer2023_own_study.xlsx'
    def matrix(sheet):
        rows = read_catalog(workbook, sheet=sheet)
        if len(rows) != 11 or any(set(r['values']) != set('ABCDEFGHIJK') for r in rows):
            raise ValueError('Expected complete 11 conditions x 11 anonymous participants')
        array = np.array([[float(r['values'][c]) for c in 'ABCDEFGHIJK'] for r in rows])
        if not np.isfinite(array).all():
            raise ValueError('Nonfinite author data')
        return array
    with zipfile.ZipFile(workbook) as z:
        ns = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
        names = [s.attrib['name'] for s in ET.fromstring(z.read('xl/workbook.xml')).findall('m:sheets/m:sheet', ns)]
        if names[:4] != ['Muscle_Force_Individual', 'Vicon_angular_Velocity', 'Velocity_Individual', 'Isomed_angular_Velocity']:
            raise ValueError('Author sheet mapping changed')
    reference = matrix(1)
    actual = -matrix(2)  # Author plotting code negates dorsiflexion-positive Vicon data.
    dynamometer = matrix(4)
    preset = np.broadcast_to(np.arange(0,201,20)[:,None], actual.shape)
    if not (reference[0] == 1).all() or not (actual[0] == 0).all() or (actual < 0).any():
        raise ValueError('Expected zero-speed unit normalization and shortening-only velocities')
    profile = load_profile(); joint_names = list(profile['joints']); i = joint_names.index('L_Ankle_y')
    model = HumanJointModel(profile, joint_names, dtype=torch.float64, features=('strength',), strength_cohort='male')
    predictions = {}; masks = {}
    for name, speed in [('preset',preset), ('dynamometer',dynamometer), ('camera',actual)]:
        q = torch.zeros((*speed.shape, len(joint_names)), dtype=torch.float64)
        qd = torch.zeros_like(q); qd[...,i] = torch.tensor(np.deg2rad(speed))
        _, cap, outside = model.strength_caps(q, qd, directional_domain=True)
        predictions[name] = (cap[...,i]/cap[0,:,i]).numpy()
        masks[name] = ~outside[...,i,1].numpy()
    common = np.logical_and.reduce(list(masks.values())); common[0] = False
    subjects = []
    for j in range(11):
        valid = common[:,j]
        if valid.sum() < 3:
            raise ValueError('Insufficient common source-domain support')
        subjects.append({'column': chr(65+j), 'evaluation_rows': (np.flatnonzero(valid)+1).tolist(),
                         'metrics': {name: curve_metrics(pred[valid,j], reference[valid,j], scale=1.)
                                     for name,pred in predictions.items()}})
    result = {'status':'candidate', 'scope':'Within-study normalized-force sensitivity to velocity measurement; not absolute Nm or independent confirmation',
              'normalization':'Author individual zero-speed force; model individual zero-speed cap; excluded from evaluation',
              'subjects':subjects,
              'subject_macro_nrmse':{name:float(np.mean([s['metrics'][name]['nrmse'] for s in subjects])) for name in predictions},
              'preset_deg_s':preset.tolist(), 'camera_deg_s':actual.tolist(), 'dynamometer_deg_s':dynamometer.tolist(),
              'reference_force_ratio':reference.tolist(), 'predicted_ratios':{k:v.tolist() for k,v in predictions.items()},
              'common_support':common.tolist(),
              'source_sha256':hashlib.sha256(workbook.read_bytes()).hexdigest(),
              'source_commit':'e4860a19b37e2e38514303409051027c95aa5794',
              'source_url':'https://github.com/mjhmilla/PlantarFlexorDynamometrySimulation',
              'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'limitations':['Normalized muscle force is not a measured absolute joint torque',
                             'Ratio comparison assumes constant force-to-torque mapping at neutral angle',
                             'Already-screened study; no untouched confirmation',
                             'Preload and tendon state absent in current model'],
              'runtime_adoption':False}
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'path':str(target),'common_points':int(common.sum()),'macro':result['subject_macro_nrmse']}))


def audit_holzer_isometric_traces(output):
    """Inventory untriggered force rises, never infer neural activation constants."""
    output = Path(output)
    target = output/'evaluation/screening/healthy_adult_v1_holzer_isometric_trace_audit_seed_none.json'
    if target.exists():
        raise FileExistsError(target)
    source = output/'scratch/holzer2023_isos.mat'
    author_code = output/'scratch/holzer2023_activation_tuning_source.m'
    data = loadmat(source, simplify_cells=True)
    times, torques = data['timestamp_analogs'], data['torque']
    if set(times) != set(torques):
        raise ValueError('Time and torque trial IDs differ')
    rows = []
    for name in sorted(times):
        time = np.asarray(times[name]); torque = np.asarray(torques[name])
        if time.ndim != 1 or torque.shape != time.shape or len(time) < 1001 or not np.isfinite(time).all() or not np.isfinite(torque).all():
            raise ValueError('Invalid trace array')
        if not np.allclose(np.diff(time), .001, rtol=0, atol=1e-10):
            raise ValueError('Author first-1000-sample baseline assumes 1 kHz')
        baseline = float(torque[:1000].mean())
        peak = float(torque.max()); index = int(torque.argmax())
        if peak <= baseline or index <= 1000:
            raise ValueError('No usable post-baseline positive torque rise')
        # Record onset is NOT excitation onset. Only the crossing difference is
        # meaningful here; no onset latency or model fit is reported.
        rise = observed_transition_time(time[:index+1], torque[:index+1], initial=baseline, target=peak)
        rows.append({'trial':name, 'samples':len(time), 'duration_s':float(time[-1]-time[0]),
                     'baseline_first_1000_samples_nm':baseline, 'raw_peak_nm':peak,
                     'baseline_sd_nm':float(torque[:1000].std(ddof=1)),
                     'peak_time_from_record_start_s':float(time[index]-time[0]),
                     'empirical_torque_rise_10_90_s':rise,
                     'excitation_onset_s':None,
                     'activation_time_constant_s':None})
    result = {'status':'candidate','scope':'Descriptive untriggered isometric torque traces, not biological activation validation',
              'source_sha256':{str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,author_code)},
              'source_commit':'e4860a19b37e2e38514303409051027c95aa5794','trials':rows,
              'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'processing':'No filtering; first 1000 samples baseline following author tuning code; full-trial raw peak; first threshold crossings',
              'limitations':['No measured excitation or independent onset clock',
                             'Subject identity/cohort and relation to the 11-person workbook unverified',
                             'Passive/gravity correction of these raw traces unverified',
                             'Force rise combines voluntary command, activation and tendon dynamics; cannot identify one component'],
              'parameter_fit_eligible':False,'runtime_adoption':False}
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    rises = [r['empirical_torque_rise_10_90_s'] for r in rows if r['empirical_torque_rise_10_90_s'] is not None]
    print(json.dumps({'path':str(target),'trials':len(rows),'measured_rises':len(rises),'rise_range_s':[min(rises),max(rises)] if rises else None}))


def fit_silder(records, *, selected=None, scales=None, slope_scale=1.0):
    """Nonnegative energy amplitudes; existing slopes, plus a GAS coupling basis.

    GAS exponent signs follow restoring energy, then amplitude is fitted to
    digitized curves. It is NOT a faithful transcription of the printed equation.
    """
    if not math.isfinite(slope_scale) or slope_scale<=0:
        raise ValueError('passive slope scale must be positive and finite')
    profile=load_profile(); names=list(profile['joints'])
    components=[copy.deepcopy(c) for c in profile['passive_exponentials'] if c['name'].startswith('L_')]
    components.append({'name':'L_gastrocnemius','coeff_per_rad':{'L_Knee_y':-4.5,'L_Ankle_y':-4.7},
                       'offset':0.,'energy_scale_j':1.,'source':'SILDER2007_CURVE_FIT',
                       'note':'Restoring-energy sign hypothesis, amplitude fitted to digitized source curves; not printed Eq.A.6/A.7 reproduction.'})
    for component in components:
        component['coeff_per_rad']={name:value*slope_scale for name,value in component['coeff_per_rad'].items()}
    screening_fit = selected is not None
    if selected is None:
        selected=[r for r in records if r['reference']=='Silder et al. (2007)' and r['status']=='candidate']
        if len(selected)!=12:
            raise ValueError(f'Expected 12 Silder sagittal curves, found {len(selected)}')
    if not selected:
        raise ValueError('No calibration curves')
    a=np.array([[c['coeff_per_rad'].get(n,0) for n in names] for c in components])
    offset=np.array([c['offset'] for c in components])
    design=[]; targets=[]
    # Use joint-level calibration ranges (not candidate output) and equal total
    # weight for each joint, then each curve within a joint.
    if scales is None:
        scales={joint:float(np.ptp(np.concatenate([r['torque_nm'] for r in selected if r['joint']==joint])))
                for joint in set(r['joint'] for r in selected)}
    for r in selected:
        q=np.array(r['q_rad']); i=names.index(r['joint']); y=np.array(r['torque_nm'])
        basis=-np.exp(np.minimum(q@a.T+offset,math.log(profile['numerics']['exponential_force_cap_nm'])))*a[:,i]
        studies={s['reference'] for s in selected if s['joint']==r['joint']}
        count=len(studies)*sum(s['joint']==r['joint'] and s['reference']==r['reference'] for s in selected)
        weight=1/(scales[r['joint']]*math.sqrt(len(y)*count))
        design.append(basis*weight);targets.append(y*weight)
    coefficients,residual=nnls(np.concatenate(design),np.concatenate(targets))
    for c,scale in zip(components,coefficients): c['energy_scale_j']=float(scale)
    fitted_profile=copy.deepcopy(profile)
    fitted_profile['passive_exponentials']=[]
    for side in ('L','R'):
        for c in components:
            r=copy.deepcopy(c);r['name']=side+c['name'][1:]
            r['coeff_per_rad']={side+n[1:]:v for n,v in c['coeff_per_rad'].items()}
            fitted_profile['passive_exponentials'].append(r)
    model=HumanJointModel(fitted_profile,names,dtype=torch.float64)
    baseline=HumanJointModel(profile,names,dtype=torch.float64)
    comparison=[]
    for r in selected:
        q=torch.tensor(r['q_rad'],dtype=torch.float64);i=names.index(r['joint'])
        ref=r['torque_nm'];scale=scales[r['joint']]
        b=curve_metrics(baseline.elastic_torque(q)[:,i].numpy(),ref,scale=scale)
        c=curve_metrics(model.elastic_torque(q)[:,i].numpy(),ref,scale=scale)
        comparison.append({'id':r['id'],'joint':r['joint'],'baseline':b,'candidate':c,
                           'fit_target_gate':contract_gate(c['nrmse'],'passive_curve_nrmse',evidence_kind='digitized_fitted_curve',required_evidence='digitized_fitted_curve'),
                           'independent_biological_gate':contract_gate(c['nrmse'],'passive_curve_nrmse',evidence_kind='digitized_fitted_curve')})
    return {'implementation_id':'healthy_adult_v1','status':'candidate','source':'https://doi.org/10.6084/m9.figshare.26018563',
            'source_scope':('Screening study aggregates plus Silder calibration; no independent biological confirmation'
                            if screening_fit else 'Silder2007 figures, digitized fitted curves; calibration only, no independent biological confirmation'),
            'fit_method':'nonnegative least squares for conservative energy amplitudes, fixed supplied slope scale; equal joint then study then curve weights',
            'slope_scale':slope_scale,
            'calibration_curve_ids':[r['id'] for r in selected], 'calibration_scales_nm':scales,
            'weighted_residual':float(residual),'passive_exponentials':fitted_profile['passive_exponentials'],'comparison':comparison}


def main(source,output):
    source,output=Path(source),Path(output)
    records=import_curves(source/'chen2025_datasets.mat',source/'chen2025_catalog.xlsx')
    paths=[source/'chen2025_datasets.mat',source/'chen2025_catalog.xlsx']
    provenance={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    registry={'source':'https://doi.org/10.6084/m9.figshare.26018563','sha256':provenance,'records':records}
    target=output/'configs'/'curve_selection_registry.json'
    if target.exists(): raise FileExistsError(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(registry,indent=2,allow_nan=False)+'\n')
    fitted=fit_silder(records);fitted['source_sha256']=provenance
    fit_path=output/'evaluation'/'screening'/'healthy_adult_v1_passive_curve_fit_seed_none.json'
    fit_path.write_text(json.dumps(fitted,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'fit_path':str(fit_path),'curves':len(records),'eligible':sum(r['status']=='candidate' for r in records),
                      'fitted_curves':len(fitted['comparison']),
                      'baseline_macro_nrmse':float(np.mean([r['baseline']['nrmse'] for r in fitted['comparison']])),
                      'candidate_macro_nrmse':float(np.mean([r['candidate']['nrmse'] for r in fitted['comparison']]))}))


def screen_passive_fit(output, *, slope_scale=1.0):
    """Study-held-out diagnostics within screening; confirmation remains untouched.

    Every fold fits energy amplitudes afresh without its held-out study. Scales
    remain those frozen from Silder calibration. These are model selection
    diagnostics, not independent confirmation or audited individual measurements.
    """
    output=Path(output)
    suffix='' if slope_scale==1.0 else f'_slope{format(slope_scale,"g").replace(".","p")}'
    path=output/'evaluation'/'screening'/f'healthy_adult_v1_passive_study_fit{suffix}_seed_none.json'
    if path.exists():
        raise FileExistsError(path)
    registry_path=output/'configs'/'curve_selection_registry.json'
    split_path=output/'configs'/'passive_study_split.json'
    registry=json.loads(registry_path.read_text())
    split=json.loads(split_path.read_text())['study_split']
    previous_path=Path(__file__).with_name('candidate_parameters.json')
    previous=json.loads(previous_path.read_text())
    scales=previous['passive_fit']['calibration_scales_nm']
    profile=load_profile(); names=list(profile['joints'])
    baseline=HumanJointModel(profile,names,dtype=torch.float64)
    calibration=[]; screening=[]; exclusions=[]
    for original in registry['records']:
        # Do not inspect confirmation torque/angle values during model selection.
        assignment=split.get(original['reference'])
        if assignment not in ('calibration','screening'):
            continue
        if original['status']!='candidate' or original['measure_type']!='Passive':
            continue
        r=copy.deepcopy(original)
        q=np.array(r['q_rad'])
        mask=((q>=baseline.lower.numpy()-1e-8)&(q<=baseline.upper.numpy()+1e-8)).all(1)
        if mask.sum()<3:
            exclusions.append({'id':r['id'],'reason':'fewer_than_three_in_rom_points'})
            continue
        r['q_rad']=q[mask].tolist()
        r['torque_nm']=np.array(r['torque_nm'])[mask].tolist()
        (calibration if assignment=='calibration' else screening).append(r)
    studies=sorted({r['reference'] for r in screening})
    if len(studies)<3 or not calibration:
        raise ValueError('Need calibration and at least three screening studies')
    folds=[]
    for held_out in studies:
        training=calibration+[r for r in screening if r['reference']!=held_out]
        fitted=fit_silder([],selected=training,scales=scales,slope_scale=slope_scale)
        fitted_profile=copy.deepcopy(profile)
        fitted_profile['passive_exponentials']=fitted['passive_exponentials']
        candidate=HumanJointModel(fitted_profile,names,dtype=torch.float64)
        curves=[]
        for r in screening:
            if r['reference']!=held_out:
                continue
            q=torch.tensor(r['q_rad'],dtype=torch.float64); i=names.index(r['joint'])
            curves.append({'id':r['id'],'joint':r['joint'],
                'baseline':curve_metrics(baseline.elastic_torque(q)[:,i].numpy(),r['torque_nm'],scale=scales[r['joint']]),
                'candidate':curve_metrics(candidate.elastic_torque(q)[:,i].numpy(),r['torque_nm'],scale=scales[r['joint']])})
        folds.append({'held_out_study':held_out,'training_curve_ids':fitted['calibration_curve_ids'],
                      'passive_exponentials':fitted['passive_exponentials'],'curves':curves,
                      'baseline_nrmse':float(np.mean([r['baseline']['nrmse'] for r in curves])),
                      'candidate_nrmse':float(np.mean([r['candidate']['nrmse'] for r in curves]))})
    fitted=fit_silder([],selected=calibration+screening,scales=scales,slope_scale=slope_scale)
    result={'implementation_id':'healthy_adult_v1','status':'candidate',
            'slope_scale':slope_scale,
            'scope':'leave-one-study-out screening only; not biological confirmation',
            'selection_policy':'All eligible passive screening curves with >=3 in-ROM points; no error-dependent exclusions. Metadata uncertainty remains unaudited.',
            'source_sha256':registry['sha256'],
            'split_sha256':hashlib.sha256(split_path.read_bytes()).hexdigest(),
            'previous_parameters_sha256':hashlib.sha256(previous_path.read_bytes()).hexdigest(),
            'calibration_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'folds':folds,'excluded':exclusions,'fitted':fitted,
            'baseline_study_macro_nrmse':float(np.mean([f['baseline_nrmse'] for f in folds])),
            'candidate_study_macro_nrmse':float(np.mean([f['candidate_nrmse'] for f in folds]))}
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'path':str(path),'studies':len(folds),
                      'baseline':result['baseline_study_macro_nrmse'],
                      'candidate':result['candidate_study_macro_nrmse']}))


def evaluate_passive(output,stage):
    """Compare study-disjoint curves without fitting or changing any parameter.

    Digitization/pooled cohorts/metadata assumptions are explicit limitations.
    Preliminary fit-target gates are not sufficient for model adoption.
    """
    output=Path(output)
    registry_path=output/'configs'/'curve_selection_registry.json'
    split_path=output/'configs'/'passive_study_split.json'
    registry=json.loads(registry_path.read_text());split=json.loads(split_path.read_text())['study_split']
    parameters_path=Path(__file__).with_name('candidate_parameters.json')
    params=json.loads(parameters_path.read_text())
    baseline=HumanJointModel(load_profile(),list(load_profile()['joints']),dtype=torch.float64)
    candidate=HumanJointModel(load_profile(),list(load_profile()['joints']),dtype=torch.float64,features=('passive_fit',))
    results=[]
    for r in registry['records']:
        if r['status']!='candidate' or r['measure_type']!='Passive' or split.get(r['reference'])!=stage:
            continue
        q=torch.tensor(r['q_rad'],dtype=torch.float64);i=baseline.names.index(r['joint'])
        ref=np.array(r['torque_nm']);scale=params['passive_fit']['calibration_scales_nm'][r['joint']]
        pred=candidate.elastic_torque(q)[:,i].numpy();original=baseline.elastic_torque(q)[:,i].numpy()
        in_rom=((q>=baseline.lower-1e-8)&(q<=baseline.upper+1e-8)).all(-1).numpy()
        item={'id':r['id'],'study':r['reference'],'doi':r['doi'],'subject':r['subject'],'joint':r['joint'],
              'evidence_kind':r['evidence_kind'],'metadata':r['metadata'],
              'rom_covered_fraction':float(in_rom.mean()),'points':len(ref),
              'baseline_full_curve':curve_metrics(original,ref,scale=scale),
              'candidate_full_curve':curve_metrics(pred,ref,scale=scale)}
        if in_rom.sum()>=3:
            item['baseline_in_rom']=curve_metrics(original[in_rom],ref[in_rom],scale=scale)
            item['candidate_in_rom']=curve_metrics(pred[in_rom],ref[in_rom],scale=scale)
            item['comparison_target_gate']=contract_gate(item['candidate_in_rom']['nrmse'],'passive_curve_nrmse',
                evidence_kind=r['evidence_kind'],required_evidence='digitized_measurement')
        else:
            item['comparison_target_gate']=contract_gate(None,'passive_curve_nrmse',evidence_kind=r['evidence_kind'])
        item['independent_biological_gate']={
            'status':'insufficient_evidence',
            'reason':'Digitized study aggregates require original protocol/metadata-uncertainty audit and subject/cohort compatibility; no raw held-out subject data.'}
        results.append(item)
    grouped={}
    for r in results:
        if 'candidate_in_rom' not in r: continue
        grouped.setdefault(r['study'],[]).append(r)
    means={study:{'baseline_nrmse':float(np.mean([r['baseline_in_rom']['nrmse'] for r in rows])),
                  'candidate_nrmse':float(np.mean([r['candidate_in_rom']['nrmse'] for r in rows]))}
           for study,rows in grouped.items()}
    result={'implementation_id':'healthy_adult_v1','features':['passive_fit'],'stage':stage,
            'status':'candidate_failed_gate','scope':'study-disjoint published aggregate curve comparison, not individual biological validation or walking',
            'source_sha256':registry['sha256'],'split_sha256':hashlib.sha256(split_path.read_bytes()).hexdigest(),
            'parameters_sha256':hashlib.sha256(parameters_path.read_bytes()).hexdigest(),
            'studies':means,'curves':results}
    path=output/'evaluation'/stage/f'healthy_adult_v1_passive_study_comparison_seed_none.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'path':str(path),'studies':len(means),'curves':len(results),'study_macro':{
        key:float(np.mean([r[key] for r in means.values()])) for key in ['baseline_nrmse','candidate_nrmse']}}))


def audit_passive_residuals(output):
    """Diagnose existing screening errors without fitting or revising their gates."""
    output = Path(output)
    path = output/'evaluation/screening/healthy_adult_v1_passive_residual_audit_seed_none.json'
    if path.exists():
        raise FileExistsError(path)
    registry_path = output/'configs/curve_selection_registry.json'
    previous_path = output/'evaluation/screening/healthy_adult_v1_passive_study_comparison_seed_none.json'
    parameters_path = Path(__file__).with_name('candidate_parameters.json')
    records = {r['id']: r for r in json.loads(registry_path.read_text())['records']}
    previous = json.loads(previous_path.read_text())
    scales = json.loads(parameters_path.read_text())['passive_fit']['calibration_scales_nm']
    profile = load_profile()
    models = {'baseline': HumanJointModel(profile, list(profile['joints']), dtype=torch.float64),
              'candidate': HumanJointModel(profile, list(profile['joints']), dtype=torch.float64, features=('passive_fit',))}
    rows = []
    for old in previous['curves']:
        if 'candidate_in_rom' not in old:
            continue
        record = records[old['id']]
        q = torch.tensor(record['q_rad'], dtype=torch.float64)
        model = models['baseline']
        mask = ((q >= model.lower-1e-8) & (q <= model.upper+1e-8)).all(-1).numpy()
        ref = np.asarray(record['torque_nm'])[mask]
        i = model.names.index(record['joint'])
        row = {'id': old['id'], 'study': old['study'], 'joint': record['joint'],
               'points': int(mask.sum()), 'original_comparison_status': old['comparison_target_gate']['status']}
        for name, model in models.items():
            pred = model.elastic_torque(q)[:, i].numpy()[mask]
            diagnostic = residual_decomposition(pred, ref, scale=scales[record['joint']])
            if not np.isclose(diagnostic['absolute']['nrmse'], old[name+'_in_rom']['nrmse'], rtol=1e-10, atol=1e-12):
                raise ValueError(f"Original screening result no longer reproduced: {old['id']} {name}")
            row[name] = diagnostic
        rows.append(row)
    result = {'status': 'candidate', 'scope': 'Screening residual diagnosis; no parameter changes or new biological passes',
              'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (registry_path, previous_path, parameters_path)},
              'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'curves': rows}
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'path': str(path), 'curves': len(rows),
                      'clark_franz': next((r for r in rows if r['id']=='isometric_passive_281_curve0'), None)}))


def fit_strength_coupling(output):
    """Estimate knee dependence from matched ankle angles within screening cohorts.

    Center log torque within dataset/ankle-angle groups. This removes arbitrary
    between-cohort strength amplitudes and the ankle-angle curve from the fit.
    It does not identify individual muscle forces or dynamic coupling.
    """
    output=Path(output)
    path=output/'evaluation'/'screening'/'healthy_adult_v1_strength_coupling_fit_seed_none.json'
    if path.exists(): raise FileExistsError(path)
    registry_path=output/'configs'/'curve_selection_registry.json'
    split_path=output/'configs'/'strength_study_split.json'
    registry=json.loads(registry_path.read_text());split=json.loads(split_path.read_text())['study_split']
    groups={}
    for r in registry['records']:
        if split.get(r['reference'])!='screening' or r['status']!='candidate' or r['measure_type']!='Active' or r['object']!='Ankle plantarflexion':
            continue
        for ankle,knee,torque in zip(r['primary_angle_deg'],r['secondary_angle_deg'],r['torque_nm']):
            if torque<=0: continue
            groups.setdefault((r['reference'],r['dataset_number'],round(ankle,6)),[]).append((math.radians(knee),math.log(torque),r['id']))
    studies={}
    for (study,number,angle),points in groups.items():
        # Duplicate observations of the same knee angle are averaged before
        # fitting; they must not receive extra weight through curve overlap.
        by_knee={}
        for knee,log_torque,curve_id in points:
            by_knee.setdefault(knee,[]).append(log_torque)
        if len(by_knee)<2: continue
        x=np.array(sorted(by_knee)); y=np.array([np.mean(by_knee[k]) for k in x])
        if np.ptp(x)<math.radians(15): continue
        studies.setdefault(study,[]).append({'dataset_number':number,'ankle_deg':angle,
            'knee_rad':x.tolist(),'log_torque':y.tolist(),
            'centered_knee':(x-x.mean()).tolist(),'centered_log_torque':(y-y.mean()).tolist(),
            'curve_ids':sorted({p[2] for p in points})})
    if len(studies)<2: raise ValueError('Need at least two study-disjoint matched-angle cohorts')
    def fit(selected):
        numerator=denominator=0.
        for study in selected:
            for group in studies[study]:
                x=np.array(group['centered_knee']); y=np.array(group['centered_log_torque'])
                weight=1/(len(studies[study])*len(x))
                numerator+=weight*float(x@y);denominator+=weight*float(x@x)
        return max(0.,-numerator/denominator)
    folds=[]
    for study,groups in studies.items():
        beta=fit([s for s in studies if s!=study])
        errors=[];baseline=[]
        for g in groups:
            x=np.array(g['centered_knee']);y=np.array(g['centered_log_torque'])
            errors.extend((-beta*x-y).tolist());baseline.extend((-y).tolist())
        folds.append({'held_out_study':study,'training_studies':[s for s in studies if s!=study],
                      'beta_per_rad':beta,'baseline_centered_log_rmse':float(np.sqrt(np.mean(np.square(baseline)))),
                      'candidate_centered_log_rmse':float(np.sqrt(np.mean(np.square(errors))))})
    result={'implementation_id':'healthy_adult_v1','status':'candidate',
            'scope':'Screening matched-angle isometric torque ratios; not biological confirmation or individual muscle identification',
            'source_sha256':registry['sha256'],'split_sha256':hashlib.sha256(split_path.read_bytes()).hexdigest(),
            'fit_method':'Equal study and matched-angle group weighted centered log torque regression; beta nonnegative',
            'beta_per_rad':fit(list(studies)),'reference_knee_rad':math.radians(50),
            'knee_domain_rad':[min(k for gs in studies.values() for g in gs for k in g['knee_rad']),
                               max(k for gs in studies.values() for g in gs for k in g['knee_rad'])],
            'studies':studies,'folds':folds}
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'path':str(path),'beta_per_rad':result['beta_per_rad'],'folds':folds}))


def reported_strength_cohort(subject):
    """Use explicit source participant counts only; unknown/mixed is not inferred."""
    sexes = set(re.findall(r'\b\d+\s*([MF])\b', subject))
    return {'M':'male','F':'female'}.get(next(iter(sexes))) if len(sexes)==1 else None


def reported_strength_size(subject):
    """Extract explicit single cohort means in cm/kg; reject ranges and ambiguity."""
    mean = r'(\d+(?:\.\d+)?)(?:\s*±\s*\d+(?:\.\d+)?)?'
    heights = re.findall(r'(?:^|[;(])\s*'+mean+r'\s*cm\s*(?=[;)])',subject)
    masses = re.findall(r'(?:^|[;(])\s*'+mean+r'\s*kg\s*(?=[;)])',subject)
    if len(heights)!=1 or len(masses)!=1:
        return None
    size = (float(masses[0]),float(heights[0])/100)
    return size if all(x>0 and math.isfinite(x) for x in size) else None


def screen_isometric_strength(output, features=('strength',), *, match_cohort=False, match_size=False):
    """Out-of-source ankle strength diagnostics, never passive or total torque.

    Preserve shared study assignments across passive/active endpoints. Independent
    confirmation remains unopened. No fitting, cohort scaling, or angle offsets
    are inferred from the comparison curves.
    """
    output=Path(output)
    suffix='_coupled' if 'strength_coupling' in features else ''
    if match_cohort:
        suffix += '_cohort_matched'
    if match_size:
        suffix += '_size_matched'
    path=output/'evaluation'/'screening'/f'healthy_adult_v1_isometric_study_comparison{suffix}_seed_none.json'
    if path.exists():
        raise FileExistsError(path)
    registry_path=output/'configs'/'curve_selection_registry.json'
    registry=json.loads(registry_path.read_text())
    passive_split_path=output/'configs'/'passive_study_split.json'
    shared=json.loads(passive_split_path.read_text())['study_split']
    split_path=output/'configs'/'strength_study_split.json'
    assignments={}
    for r in registry['records']:
        if r['status']=='candidate' and r['measure_type']=='Active':
            study=r['reference']
            assignments[study]=shared.get(study, 'screening' if int(hashlib.sha256(study.encode()).hexdigest()[:8],16)%2==0 else 'confirmation')
    assignment_record={'assignment':'Preserve passive study split; otherwise SHA256(reference) first8hex parity even=screening',
                       'study_split':assignments,
                       'passive_split_sha256':hashlib.sha256(passive_split_path.read_bytes()).hexdigest()}
    if split_path.exists():
        if json.loads(split_path.read_text())!=assignment_record:
            raise ValueError('Frozen strength study split changed')
    else:
        split_path.write_text(json.dumps(assignment_record,indent=2)+'\n')
    parameters_path=Path(__file__).with_name('candidate_parameters.json')
    params=json.loads(parameters_path.read_text())['strength']
    scales={}
    for direction in ('plantarflexion','dorsiflexion'):
        scales[direction]=float(np.mean([c[0]*cohort['mass_kg']*cohort['height_m']*params['gravity_m_s2']
            for c,cohort in zip(params['directions'][direction]['coefficients'],params['cohorts'])]))
    profile=load_profile();names=list(profile['joints'])
    baseline=HumanJointModel(profile,names,dtype=torch.float64)
    candidate=HumanJointModel(profile,names,dtype=torch.float64,features=features)
    cohort_models={sex:HumanJointModel(profile,names,dtype=torch.float64,features=features,strength_cohort=sex)
                   for sex in ('male','female')} if match_cohort else {}
    curves=[]
    for r in registry['records']:
        if r['status']!='candidate' or r['measure_type']!='Active' or assignments.get(r['reference'])!='screening':
            continue
        if r['object'] not in ('Ankle plantarflexion','Ankle dorsiflexion'):
            raise ValueError('Strength diagnostic needs an explicit direction mapping')
        direction='plantarflexion' if r['object']=='Ankle plantarflexion' else 'dorsiflexion'
        sign=params['directions'][direction]['sign']
        q=torch.tensor(r['q_rad'],dtype=torch.float64);i=names.index(r['joint'])
        reported_cohort=reported_strength_cohort(r['subject'])
        selected_model=cohort_models.get(reported_cohort,candidate)
        reported_size=reported_strength_size(r['subject'])
        if match_size and reported_cohort is not None and reported_size is not None:
            selected_model=HumanJointModel(profile,names,dtype=torch.float64,features=features,
                strength_cohort=selected_model.strength_cohort,strength_reference_size=reported_size)
        negative,positive,outside=selected_model.strength_caps(q,torch.zeros_like(q))
        pred=(positive[:,i] if sign>0 else -negative[:,i]).numpy()
        base=np.full(len(q),float(baseline.positive[i] if sign>0 else -baseline.negative[i]))
        ref=np.array(r['torque_nm'])
        if np.any(ref*sign<0):
            raise ValueError('Reference torque sign conflicts with anatomical direction')
        valid=((q>=baseline.lower-1e-8)&(q<=baseline.upper+1e-8)).all(-1).numpy() & ~outside[:,i].numpy()
        item={'id':r['id'],'study':r['reference'],'direction':direction,'subject':r['subject'],
              'doi':r['doi'],'metadata':r['metadata'],'points':len(q),'valid_points':int(valid.sum()),
              'normalization_nm':scales[direction],
              'normalization_basis':'Anderson young cohort mean C1 * body weight * height; frozen source amplitude, not comparison peak',
              'knee_angle_deg':r['secondary_angle_deg'],
              'source_strength_cohort':selected_model.strength_cohort,
              'reported_single_sex_cohort':reported_cohort,
              'source_sex_matched':reported_cohort is not None and selected_model.strength_cohort==reported_cohort,
              'reported_reference_size_kg_m':reported_size,
              'selected_reference_size_kg_m':selected_model.strength_reference_size,
              'source_size_matched':selected_model.strength_reference_size is not None,
              'remaining_mismatch':'Source age, apparatus and individual strength remain unmatched; reference size scaling changes no body geometry or inertia.',
              'source_knee_protocol_match':bool(np.allclose(r['secondary_angle_deg'],50,rtol=0,atol=1e-6)),
              'independent_biological_gate':{'status':'insufficient_evidence',
                  'reason':'Aggregate cohort mismatch, unmodeled adjacent-joint dependence and unaudited protocol/uncertainty; exploratory screening only.'}}
        if valid.sum()>=3:
            item['baseline']=curve_metrics(base[valid],ref[valid],scale=scales[direction])
            item['candidate']=curve_metrics(pred[valid],ref[valid],scale=scales[direction])
            item['comparison_target_gate']=contract_gate(item['candidate']['nrmse'],'isometric_torque_nrmse',
                evidence_kind=r['evidence_kind'],required_evidence='digitized_measurement')
        else:
            item['comparison_target_gate']=contract_gate(None,'isometric_torque_nrmse',evidence_kind=r['evidence_kind'])
        curves.append(item)
    studies={}
    for r in curves:
        if 'candidate' in r:
            studies.setdefault(r['study'],[]).append(r)
    means={s:{key+'_nrmse':float(np.mean([r[key]['nrmse'] for r in rs]))
              for key in ('baseline','candidate')} for s,rs in studies.items()}
    result={'implementation_id':'healthy_adult_v1','features':features,'status':'candidate_failed_gate',
            'match_source_sex_cohort':match_cohort,
            'match_source_reference_size':match_size,
            'scope':'Screening isometric active ankle torque only; no independent biological confirmation or walking',
            'source_sha256':registry['sha256'],'split_sha256':hashlib.sha256(split_path.read_bytes()).hexdigest(),
            'parameters_sha256':hashlib.sha256(parameters_path.read_bytes()).hexdigest(),
            'calibration_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scales_nm':scales,'studies':means,'curves':curves}
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'path':str(path),'studies':len(means),'curves':len(curves),
                      'study_macro':{key:float(np.mean([r[key] for r in means.values()]))
                                     for key in ('baseline_nrmse','candidate_nrmse')}}))


def audit_strength_consistency(output):
    """Check shared-state compatibility of already inspected screening curves."""
    output = Path(output)
    path = output/'evaluation/screening/healthy_adult_v1_strength_consistency_seed_none.json'
    if path.exists():
        raise FileExistsError(path)
    inputs = [output/'configs/curve_selection_registry.json',
              output/'evaluation/screening/healthy_adult_v1_isometric_study_comparison_seed_none.json']
    registry, comparison = [json.loads(p.read_text()) for p in inputs]
    selected = {r['id']:r for r in comparison['curves'] if 'candidate' in r}
    profile = load_profile(); names = list(profile['joints'])
    model = HumanJointModel(profile,names,dtype=torch.float64,features=('strength',))
    curves = []
    for row in registry['records']:
        if row['id'] not in selected:
            continue  # confirmation records are never processed
        previous = selected[row['id']]
        q = torch.tensor(row['q_rad'],dtype=torch.float64)
        i = names.index(row['joint'])
        outside = model.strength_caps(q,torch.zeros_like(q))[2][:,i]
        valid = ((q>=model.lower-1e-8)&(q<=model.upper+1e-8)).all(-1)&~outside
        if int(valid.sum()) != previous['valid_points']:
            raise ValueError('Screening support changed; audit cannot silently change conditions')
        curves.append((row, previous, q[valid].numpy(),np.asarray(row['torque_nm'])[valid.numpy()]))
    pairs = []
    for j,(a,pa,qa,ya) in enumerate(curves):
        for b,pb,qb,yb in curves[j+1:]:
            if a['reference'] == b['reference'] or a['joint'] != b['joint'] or pa['direction'] != pb['direction']:
                continue
            if pa['normalization_nm'] != pb['normalization_nm']:
                raise ValueError('Shared normalization required')
            bound = shared_state_error_bound(qa,ya,qb,yb,scale=pa['normalization_nm'])
            if bound['shared_samples']:
                threshold_check = contract_gate(bound['minimum_possible_max_curve_nrmse'],
                    'isometric_torque_nrmse', evidence_kind='mathematical_bound', required_evidence='mathematical_bound')
                pairs.append(dict(bound,curve_a=a['id'],curve_b=b['id'],study_a=a['reference'],study_b=b['reference'],
                                  subject_a=a['subject'],subject_b=b['subject'],
                                  comparison_threshold=threshold_check['threshold'],
                                  threshold_contract_sha256=threshold_check['contract_sha256'],
                                  target_simultaneously_impossible=threshold_check['status']=='fail'))
    result = {'status':'candidate','scope':'Same-state deterministic strength compatibility; no data rejection or threshold change',
              'assumptions':'Exact identical full joint coordinates, zero velocity, same maximal activation, no subject-specific input',
              'source_sha256':{str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
              'implementation_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (Path(__file__),Path(__file__).with_name('metrics.py'),
                            Path(__file__).with_name('dynamics.py'),Path(__file__).with_name('candidate_parameters.json'),
                            Path(__file__).with_name('healthy_adult_v1.json'))},
              'pairs':pairs,'incompatible_pair_count':sum(p['target_simultaneously_impossible'] for p in pairs)}
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'path':str(path),'pairs':len(pairs),'incompatible_pairs':result['incompatible_pair_count']}))


def audit_strength_anchor(output, *, verify_runtime=False):
    """Neutral-angle amplitude calibration, other angles of screening curves only."""
    output=Path(output)
    suffix='_runtime' if verify_runtime else ''
    path=output/f'evaluation/screening/healthy_adult_v1_strength_neutral_anchor{suffix}_seed_none.json'
    if path.exists():
        raise FileExistsError(path)
    previous_path=output/'evaluation/screening/healthy_adult_v1_isometric_study_comparison_coupled_cohort_matched_size_matched_seed_none.json'
    previous=json.loads(previous_path.read_text())
    selected={r['id']:r for r in previous['curves'] if 'candidate' in r}
    registry_path=output/'configs/curve_selection_registry.json'
    registry=json.loads(registry_path.read_text())
    profile=load_profile();names=list(profile['joints']);curves=[]
    for row in registry['records']:
        if row['id'] not in selected:
            continue
        old=selected[row['id']]
        model=HumanJointModel(profile,names,dtype=torch.float64,features=previous['features'],
                             strength_cohort=old['source_strength_cohort'],strength_reference_size=old['selected_reference_size_kg_m'])
        q=torch.tensor(row['q_rad'],dtype=torch.float64); i=names.index(row['joint'])
        neg,pos,outside=model.strength_caps(q,q*0)
        valid=((q>=model.lower-1e-8)&(q<=model.upper+1e-8)).all(-1)&~outside[:,i]
        pred=(pos[:,i] if old['direction']=='plantarflexion' else -neg[:,i])[valid].numpy()
        ref=np.asarray(row['torque_nm'])[valid.numpy()]
        if len(pred)!=old['valid_points'] or not np.isclose(curve_metrics(pred,ref,scale=old['normalization_nm'])['nrmse'],old['candidate']['nrmse'],rtol=1e-10,atol=1e-12):
            raise ValueError('Previous candidate conditions or prediction changed')
        anchors=np.flatnonzero(np.asarray(row['primary_angle_deg'])[valid.numpy()]==0)
        item={'id':row['id'],'study':row['reference'],'subject':row['subject'],
              'valid_original_indices':np.flatnonzero(valid.numpy()).tolist(),
              'normalization_nm':old['normalization_nm']}
        if len(anchors)!=1:
            item.update(status='not_evaluated',reason='Requires exactly one observed neutral-angle sample; no interpolation or closest-angle selection')
        else:
            try:
                result=anchored_curve_metrics(pred,ref,anchor_index=int(anchors[0]),scale=old['normalization_nm'])
            except ValueError as error:
                item.update(status='not_evaluated',reason=str(error))
            else:
                item.update(result,status='candidate',
                            target_gate=contract_gate(result['anchored']['nrmse'],'isometric_torque_nrmse',evidence_kind='within_curve_screening',required_evidence='within_curve_screening'))
                if verify_runtime:
                    sign=1 if old['direction']=='plantarflexion' else -1
                    factors=[1.,1.];factors[int(sign>0)]=result['amplitude_factor']
                    calibrated=HumanJointModel(profile,names,dtype=torch.float64,features=previous['features'],
                        strength_cohort=old['source_strength_cohort'],strength_reference_size=old['selected_reference_size_kg_m'],
                        active_strength_scale={row['joint']:factors})
                    total,active,_=calibrated.torques(torch.full_like(q,sign*1e6),q,q*0,dt=.001)
                    measured=active[:,i][valid].numpy()
                    error=float(np.max(np.abs(measured-result['amplitude_factor']*pred)))
                    item['runtime_scale_verification']={
                        'active_strength_scale':{row['joint']:factors},
                        'torque_agreement_gate':contract_gate(error,'torque_sum_max_nm',evidence_kind='numerical',required_evidence='numerical'),
                        'backend_ceiling_excess_nm':float((total.abs()-calibrated.backend_limit).clamp_min(0).max()),
                        'heldout_metrics':curve_metrics(measured[result['evaluation_indices']],ref[result['evaluation_indices']],scale=old['normalization_nm'])}
        curves.append(item)
    studies={}
    for row in curves:
        if 'anchored' in row:
            studies.setdefault(row['study'],[]).append(row)
    means={study:{key+'_nrmse':float(np.mean([row[key]['nrmse'] for row in rows])) for key in ('unscaled','anchored')}
           for study,rows in studies.items()}
    result={'status':'candidate','scope':'Within-cohort screening shape diagnostic, not independent biological confirmation',
            'anchor_rule':'Exactly one observed primary angle 0 degrees per curve; evaluate all other valid angles',
            'source_sha256':{str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (previous_path,registry_path)},
            'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'curves':curves,'studies':means}
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'path':str(path),'evaluated_studies':len(means),
        'study_macro':{key:float(np.mean([r[key] for r in means.values()])) for key in ('unscaled_nrmse','anchored_nrmse')} if means else {}}))


def screen_looft_shoulder_fatigue(output):
    """Screen fixed literature multipliers with explicit MVC-probe assumptions."""
    from .dynamics import fatigue_compartment_step
    output = Path(output)
    path = output/'evaluation/screening/healthy_adult_v1_looft2020_shoulder_screen_seed_none.json'
    if path.exists():
        raise FileExistsError(path)
    source = output/'evaluation/screening/healthy_adult_v1_looft2020_shoulder_reference_seed_none.json'
    reference = json.loads(source.read_text())
    parameter_path = Path(__file__).with_name('candidate_parameters.json')
    parameters = json.loads(parameter_path.read_text())['fatigue']
    rates = parameters['regions']['Shoulder']
    rows = []
    dt = .1
    for task, load, on_steps in [('50:50', .5, 50), ('70:70', .7, 70)]:
        observations = [r for r in reference['rows'] if r['task'] == task]
        probe_steps = {round(r['time_s']/dt):r for r in observations}
        duration_steps = max(probe_steps)+30
        for multiplier in [1., 15., 30.]:
            # Batch 0: no probe, capacity at nominal observation time.
            # Batch 1: replace the next 3 s with MVC, maintain wall-clock cycle.
            state = torch.zeros(2, 3, dtype=torch.float64)
            state[:, 0] = 1
            probe_start = None
            peak = 0.
            for step in range(duration_steps):
                if step in probe_steps:
                    row = probe_steps[step]
                    prediction = 100*state[0, 2].item()
                    rows.append(dict(task=task, time_s=row['time_s'], n=row['n'],
                        protocol='no_probe_capacity', rest_multiplier=multiplier,
                        predicted_decline_percent_MVC=prediction,
                        observed_decline_percent_MVC=row['observed_decline_percent_MVC']))
                    probe_start, peak = step, 0.
                targets = torch.full((2,), load if step % 100 < on_steps else 0., dtype=state.dtype)
                if probe_start is not None:
                    targets[1] = 1.
                state = fatigue_compartment_step(state, targets, dt=dt,
                    fatigue_rate=rates['fatigue_rate_per_s'], recovery_rate=rates['recovery_rate_per_s'],
                    tracking_rate=parameters['tracking_rate_per_s'], rest_multiplier=multiplier)
                if probe_start is not None:
                    peak = max(peak, state[1, 1].item())
                    if step+1 == probe_start+30:
                        row = probe_steps[probe_start]
                        rows.append(dict(task=task,time_s=row['time_s'],n=row['n'],
                            protocol='replace_next_3s_peak_active',rest_multiplier=multiplier,
                            predicted_decline_percent_MVC=100*(1-peak),
                            observed_decline_percent_MVC=row['observed_decline_percent_MVC']))
                        probe_start = None
    summaries=[]
    for protocol in ['no_probe_capacity','replace_next_3s_peak_active']:
        for multiplier in [1.,15.,30.]:
            for task in ['50:50','70:70']:
                selected=[r for r in rows if r['protocol']==protocol and r['rest_multiplier']==multiplier and r['task']==task and r['n']>=15]
                error=np.array([r['predicted_decline_percent_MVC']-r['observed_decline_percent_MVC'] for r in selected])/100
                summaries.append(dict(protocol=protocol,rest_multiplier=multiplier,task=task,
                    n_timepoints=len(selected),minimum_retained_fraction=.75,
                    nrmse_initial_MVC=float(np.sqrt(np.mean(error**2))),bias_initial_MVC=float(error.mean())))
    result=dict(status='candidate',evidence_kind='external_study_screening',
        independent_confirmation=False,runtime_adoption=False,dt_s=dt,rates=rates,
        initial_state=[1,0,0],rows=rows,summaries=summaries,
        limitations=['Probe replacement phase is an explicit assumption, not recovered individual torque traces.',
            'Later means have survivor selection; summary uses >=75% initial sample retained, separately by task.',
            'This evaluates normalized compartment outputs, not full shoulder musculoskeletal mechanics.',
            'No parameter fitted here; r1/r15/r30 are fixed literature candidates. This is screening, not confirmation.'],
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,parameter_path,Path(__file__),Path(__file__).with_name('dynamics.py'))})
    with path.open('x') as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
    print(json.dumps({'path':str(path),'summaries':summaries}))


def audit_fatigue_endurance(output):
    """Reproducible Knee 3CC vs its source calibration curve; not confirmation."""
    from .dynamics import fatigue_compartment_step
    from .metrics import endurance_time_metrics
    output = Path(output)
    path = output/'evaluation/screening/healthy_adult_v1_fatigue_endurance_full_source_seed_none.json'
    if path.exists():
        raise FileExistsError(path)
    source = output/'scratch/frey_law2010_endurance_bioc.json'
    def tables(value):
        if isinstance(value, dict):
            if value.get('infons', {}).get('type') == 'table':
                yield value
            for child in value.values():
                yield from tables(child)
        elif isinstance(value, list):
            for child in value:
                yield from tables(child)
    table = next(t for t in tables(json.loads(source.read_text())) if t['infons']['id'] == 'T2')
    coefficients = None
    for row in ET.fromstring(table['infons']['xml']).findall('.//tbody/tr'):
        cells = [''.join(c.itertext()).strip() for c in row]
        if cells[0].startswith('Exponential'):
            break
        if cells[0] == 'Knee':
            coefficients = [float(c.replace('–', '-')) for c in cells[1:3]]
    if coefficients is None:
        raise ValueError('Knee power coefficients missing in source Table2')
    parameter_path = Path(__file__).with_name('candidate_parameters.json')
    parameters = json.loads(parameter_path.read_text())['fatigue']
    rates = parameters['regions']['Knee']
    targets = torch.arange(1, 10, dtype=torch.float64)/10
    state = torch.zeros(9, 3, dtype=torch.float64)
    state[:, 0] = 1
    dt, duration = .1, 3000.
    capacity = [np.ones(9)]
    for _ in range(round(duration/dt)):
        state = fatigue_compartment_step(state, targets, dt=dt,
            fatigue_rate=rates['fatigue_rate_per_s'], recovery_rate=rates['recovery_rate_per_s'],
            tracking_rate=parameters['tracking_rate_per_s'])
        capacity.append((state[:, 0]+state[:, 1]).numpy())
    capacity = np.asarray(capacity)
    time = np.arange(len(capacity))*dt
    rows = []
    for i, target in enumerate(targets.tolist()):
        metric = endurance_time_metrics(time, capacity[:, i], target=target)
        reference = coefficients[0]*target**coefficients[1]
        prediction = metric['endurance_time_s']
        rows.append(dict(target_fraction=target, reference_s=reference, **metric,
                         error_s=None if prediction is None else prediction-reference))
    complete = all(row['censoring'] == 'none' for row in rows)
    errors = np.array([row['error_s'] for row in rows], dtype=float)
    result = dict(status='candidate', evidence_kind='source_calibration_curve',
        independent_biological_confirmation=False, runtime_adoption=False,
        region='Knee', direction_evidence='Extension only in source meta-analysis Table1',
        source_url='https://pmc.ncbi.nlm.nih.gov/articles/PMC2891087/',
        coefficients_b0_b1=coefficients, rates=rates, tracking_rate=parameters['tracking_rate_per_s'],
        initial_state=[1, 0, 0], initial_state_basis='Assay assumption; original initialization unconfirmed',
        dt_s=dt, duration_s=duration, rows=rows,
        all_endpoints_observed=complete,
        rmse_s=float(np.sqrt(np.mean(errors**2))) if complete else None,
        mean_relative_error=float(np.mean(errors/[r['reference_s'] for r in rows])) if complete else None,
        source_reported_optimization_rmse_s=6.7,
        limitations=['2010 power curves were used for 2012 parameter fitting; no independent subjects.',
                     'SSPRK2 continuous ODE implementation is not original 1Hz Matlab c2d reproduction.',
                     'Published error is a reproduction reference, not a healthy-population acceptance threshold.'],
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (source, parameter_path, Path(__file__), Path(__file__).with_name('dynamics.py'), Path(__file__).with_name('metrics.py'))})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps({'path':str(path), 'rmse_s':result['rmse_s'], 'all_endpoints_observed':complete}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source');p.add_argument('--output',required=True)
    p.add_argument('--evaluate-stage',choices=['screening','confirmation'])
    p.add_argument('--import-toe-population',action='store_true')
    p.add_argument('--fit-population-knee',action='store_true')
    p.add_argument('--screen-passive-fit',action='store_true')
    p.add_argument('--screen-isometric-strength',action='store_true')
    p.add_argument('--screen-holzer-velocity',action='store_true')
    p.add_argument('--screen-isokinetic-source',choices=['holzer','barber'])
    p.add_argument('--screen-wind-rouse-strength',action='store_true')
    p.add_argument('--audit-fatigue-endurance',action='store_true')
    p.add_argument('--screen-looft-shoulder-fatigue',action='store_true')
    p.add_argument('--audit-holzer-measured-velocity',action='store_true')
    p.add_argument('--audit-holzer-isometric-traces',action='store_true')
    p.add_argument('--fit-strength-coupling',action='store_true')
    p.add_argument('--strength-coupling',action='store_true')
    p.add_argument('--audit-strength-consistency',action='store_true')
    p.add_argument('--audit-passive-residuals',action='store_true')
    p.add_argument('--match-strength-cohort',action='store_true')
    p.add_argument('--match-strength-size',action='store_true')
    p.add_argument('--audit-strength-anchor',action='store_true')
    p.add_argument('--verify-runtime-scale',action='store_true')
    p.add_argument('--passive-slope-scale',type=float,default=1.0)
    args=p.parse_args();torch.set_num_threads(2)
    if args.import_toe_population:import_toe_population(args.source,args.output)
    elif args.fit_population_knee:fit_population_knee(args.source,args.output)
    elif args.screen_looft_shoulder_fatigue:screen_looft_shoulder_fatigue(args.output)
    elif args.audit_fatigue_endurance:audit_fatigue_endurance(args.output)
    elif args.screen_wind_rouse_strength:screen_wind_rouse_strength(args.output)
    elif args.audit_holzer_isometric_traces:audit_holzer_isometric_traces(args.output)
    elif args.audit_holzer_measured_velocity:audit_holzer_measured_velocity(args.output)
    elif args.screen_isokinetic_source:screen_isokinetic_velocity(args.output,args.screen_isokinetic_source)
    elif args.screen_holzer_velocity:screen_holzer_velocity(args.output)
    elif args.audit_passive_residuals:audit_passive_residuals(args.output)
    elif args.audit_strength_anchor:audit_strength_anchor(args.output,verify_runtime=args.verify_runtime_scale)
    elif args.audit_strength_consistency:audit_strength_consistency(args.output)
    elif args.fit_strength_coupling:fit_strength_coupling(args.output)
    elif args.screen_isometric_strength:screen_isometric_strength(args.output,('strength','strength_coupling') if args.strength_coupling else ('strength',),match_cohort=args.match_strength_cohort,match_size=args.match_strength_size)
    elif args.screen_passive_fit:screen_passive_fit(args.output,slope_scale=args.passive_slope_scale)
    elif args.evaluate_stage:evaluate_passive(args.output,args.evaluate_stage)
    elif args.source:main(args.source,args.output)
    else:p.error('--source is required for calibration')
