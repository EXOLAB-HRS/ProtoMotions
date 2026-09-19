"""Audit recorded physics-step torque use without changing a running plant."""
import torch


def summarize_trace(record, profile):
    values = record['trace'].double()
    fields = {n: values[:, i] for i, n in enumerate(record['fields'])}
    q, v = fields['q'], fields['qd']
    requested, active = fields['requested'], fields['active']
    caps = torch.where(requested >= 0, fields['positive'], fields['negative'])
    eps = 1e-6
    saturation = (requested.abs() > caps + eps) & (caps > eps)
    rows = {}
    for i, name in enumerate(record['names']):
        mask = saturation[..., i]
        run = torch.zeros(mask.shape[1], dtype=torch.long)
        longest = 0
        for sample in mask:
            run = torch.where(sample, run+1, 0)
            longest = max(longest, int(run.max()))
        demand = requested[..., i].abs() / caps[..., i].clamp_min(eps)
        rows[name] = dict(saturation_fraction=float(mask.double().mean()),
            longest_saturation_s=longest/record['physics_fps'],
            requested_peak_nm=float(requested[..., i].abs().max()),
            active_peak_nm=float(active[..., i].abs().max()),
            request_cap_ratio_p95=float(torch.quantile(demand.flatten(), .95)),
            direction_domains={})
        valid=caps[...,i]>eps
        use=active[...,i].abs()/caps[...,i].clamp_min(eps)
        total=active[...,i]+fields['passive'][...,i]
        rows[name].update(
            active_utilization_p95=float(torch.quantile(use[valid],.95)) if valid.any() else None,
            total_torque_abs_p95_nm=float(torch.quantile(total.abs().flatten(),.95)),
            net_joint_power_mean_w=float((total*v[...,i]).mean()))
        if 'kp' in record and 'kd' in record:
            kp=record['kp'].double().flatten()[i]
            kd=record['kd'].double().flatten()[i]
            if kp>eps:
                # Exact inversion for the audited proportional controller with zero target velocity.
                error=(requested[...,i]+kd*v[...,i])/kp
                rows[name]['pd_error_abs_p95_rad']=float(torch.quantile(error.abs().flatten(),.95))
                rows[name]['pd_error_abs_when_saturated_rad']=float(error[mask].abs().mean()) if mask.any() else None
                rows[name]['damping_torque_abs_p95_nm']=float(torch.quantile((kd*v[...,i]).abs().flatten(),.95))
        contacts=record.get('foot_contact_forces')
        if contacts is not None and bool(contacts.abs().max()>eps) and name.startswith(('L_','R_')):
            # Explicit diagnostic threshold, not an experimentally validated phase classifier.
            stance=contacts[...,0 if name.startswith('L_') else 1,2]>20.
            rows[name]['contact_phase']={}
            for label,selection in [('stance',stance),('swing',~stance)]:
                rows[name]['contact_phase'][label]=dict(samples=int(selection.sum()),
                    saturation_fraction=float(mask[selection].double().mean()) if selection.any() else None)
    for direction, spec in profile.get('active_strength_model', {}).get('directions', {}).items():
        for side in ('L','R'):
            name = f"{side}_{spec['joint']}"
            i = record['names'].index(name)
            angle = q[..., i]*spec['clinical_angle_sign']
            speed = v[..., i]*spec['sign']
            lo, hi = spec['angle_domain_rad']
            angle_flag = (angle < lo) | (angle > hi)
            concentric = speed > spec['concentric_limit_rad_s']
            eccentric = speed < -spec['eccentric_limit_rad_s']
            outside = angle_flag | concentric | eccentric
            selected = (requested[..., i]*spec['sign']) > 0
            rows[name]['direction_domains'][direction] = dict(
                angle_fraction=float(angle_flag.double().mean()),
                concentric_speed_fraction=float(concentric.double().mean()),
                eccentric_speed_fraction=float(eccentric.double().mean()),
                either_fraction=float(outside.double().mean()),
                selected_direction_samples=int(selected.sum()),
                selected_direction_outside_fraction=(float(outside[selected].double().mean()) if selected.any() else None))
    return dict(sample_count=int(q.shape[0]), environments=int(q.shape[1]),
                physics_fps=record['physics_fps'], checkpoint=record['checkpoint'],
                scope='PD request is not inverse-dynamics required torque. No-reset screening, including transients.',
                joints=rows)


def gait_cycle_moments(record, mass_kg=63.31, warmup_s=2., cutoff_hz=6.):
    """Screen sagittal net moments; anatomical coordinate equivalence is approximate.

    Foot-load onsets (>20N vertical, 40ms closing/opening) delimit strides.
    No per-cycle phase shift or amplitude fit to human references is allowed.
    Returns cycle-averaged waveforms per environment, matching the *type* of
    averaged human reference, not an independently validated gait event detector.
    """
    import numpy as np
    from scipy.signal import butter, sosfiltfilt
    from scipy.ndimage import binary_closing, binary_opening
    contacts=record.get('foot_contact_forces')
    if contacts is None:
        return {'status':'unavailable_no_contact_trace','waveforms':{}}
    if not bool(contacts.abs().max()>0):
        return {'status':'unavailable_zero_contact_trace','waveforms':{},'cycle_counts':{}}
    fps=record['physics_fps'];values=record['trace'].double().numpy()
    fields={n:values[:,i] for i,n in enumerate(record['fields'])}
    total=sosfiltfilt(butter(4,cutoff_hz,fs=fps,output='sos'),fields['active']+fields['passive'],axis=0)/mass_kg
    contact=contacts.numpy()[...,2]>20.
    waveforms={}; counts={}
    for si,side in enumerate(('L','R')):
        for env in range(total.shape[1]):
            stable=binary_opening(binary_closing(contact[:,env,si],structure=np.ones(max(1,round(.04*fps)))),structure=np.ones(max(1,round(.04*fps))))
            onsets=np.flatnonzero(np.diff(stable.astype(int))==1)+1
            spans=[(a,b) for a,b in zip(onsets[:-1],onsets[1:]) if a>=warmup_s*fps and .6*fps<=b-a<=2.5*fps and b<total.shape[0]-.1*fps]
            counts[f'{side}_{env}']=len(spans)
            if len(spans)<2:continue
            for joint,sign in [('Hip',-1),('Knee',-1),('Ankle',1)]:
                name=f'{side}_{joint}_y';i=record['names'].index(name)
                cycles=[np.interp(np.linspace(a,b,101),np.arange(a,b+1),total[a:b+1,env,i]*sign) for a,b in spans]
                waveforms.setdefault(joint,[]).append(dict(side=side,environment=env,cycles=len(spans),values=np.mean(cycles,axis=0).tolist()))
    return dict(status='screening_not_anatomical_validation',waveforms=waveforms,cycle_counts=counts,
                units='Nm/kg',warmup_s=warmup_s,cutoff_hz=cutoff_hz,contact_threshold_n=20.,
                mapping='WBDS sagittal Z compared to model Hip_y(-), Knee_y(-), Ankle_y(+); oblique axes/segment definitions remain a limitation')
