# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Quantitative human-model assays; SI units unless a name states otherwise.

Owned by environment_and_baselines / healthy_adult_v1 candidate validation.
No missing observation, synthetic reference, or empty trial can imply a pass.
"""

import hashlib
import json
from pathlib import Path
from .paths import v1_resource, PACKAGE_ROOT, PROTOMOTIONS_ROOT, WORKSPACE_ROOT, ASSET_ROOT

import numpy as np


def plateau_stability_metrics(time, q, qd, chapters, names):
    """Physics-rate ripple in static endpoint holds, not intentional sweeps."""
    time=_array(time,'time');q=_array(q,'q');qd=_array(qd,'qd')
    if q.shape!=qd.shape or q.shape!=(len(time),len(names)):
        raise ValueError('Time/state/name dimensions must match')
    rows=[]
    for c in chapters:
        if c['kind'] not in ('isometric','rom'):continue
        end=c['start_s']+c['duration_s'];mask=(time>=end-1)&(time<end)
        if mask.sum()<2:raise ValueError('Missing static plateau samples')
        ripple=np.rad2deg(np.ptp(q[mask],axis=0));speed=np.rad2deg(np.max(np.abs(qd[mask]),axis=0))
        j=names.index(c['joint']);worst=int(np.argmax(ripple))
        rows.append(dict(chapter=c['label'],samples=int(mask.sum()),
            focus_peak_to_peak_deg=float(ripple[j]),focus_max_speed_deg_s=float(speed[j]),
            all_joint_peak_to_peak_max_deg=float(ripple[worst]),worst_joint=names[worst],
            scope='Last 1 second, physics-rate diagnostic; not a population metric'))
    return rows


def population_assay_metrics(trace, chapters, *, pre_step=None):
    """Reduce native dynamometer traces against predeclared population records.

    References must declare source/scope separately. Passing is descriptive
    scale agreement, never independent biological validation of fitted data.
    """
    names=trace['dof_names'].tolist(); results=[]
    q=trace['q'] if pre_step is None else pre_step['q']
    qd=trace['qd'] if pre_step is None else pre_step['qd']
    if q.shape!=trace['q'].shape or qd.shape!=trace['qd'].shape:
        raise ValueError('Aligned pre-step samples must match every rendered frame')
    recovered=trace['inertia_accel']+trace['coriolis']+trace['gravity']-trace['fixture']
    for ci,c in enumerate(chapters):
        selected=trace['chapter_index']==ci; j=names.index(c['joint'])
        time=trace['time'][selected]-c['start_s']
        angle=np.rad2deg(q[selected,j])
        effort=recovered[selected,j]
        reference=c.get('population_reference')
        row={'label':c['label'],'joint':c['joint'],
             'torque_closure_max_nm':float(np.max(np.abs(effort-trace['applied'][selected,j]))),
             'state_alignment':'last pre-step state matched to effort' if pre_step is not None else 'post-step state; finite-step phase difference retained'}
        if not reference:
            row.update(status='not_assessed',reason='No protocol-aligned population record')
            results.append(row);continue
        if c['kind']=='harmonic':
            steady=(time>=3)&(time<=c['duration_s']-3)
            # Torque was sampled at the last pre-step state; retain the
            # recorded post-step error as a finite-step measurement limit.
            design=np.column_stack([q[selected,j][steady],qd[selected,j][steady],np.ones(steady.sum())])
            coefficients=np.linalg.lstsq(design,-effort[steady],rcond=None)[0]
            predicted=[float(coefficients[1])]
            row['identified_stiffness_nm_rad']=float(coefficients[0])
        elif c['kind']=='isometric':
            baseline=(time>=1)&(time<2); plateau=time>=c['duration_s']-2
            predicted=[float(c['direction']*(np.mean(effort[plateau])-np.mean(effort[baseline])))]
        elif c['kind']=='rom':
            plateau=time>=c['duration_s']-1
            predicted=[float(reference.get('clinical_sign',1)*np.mean(angle[plateau]))]
        else:
            # Compare resistance including the measured damping at the declared
            # sweep speed. Endpoint holds are excluded; no extrapolation.
            moving=(time>=2)&(time<=c['duration_s']-2)
            x=angle[moving]*reference.get('clinical_sign',1)
            y=effort[moving]*reference.get('torque_sign',1)
            order=np.argsort(x); grid=np.asarray(reference['angle_deg'])
            if grid.min()<x.min()-.1 or grid.max()>x.max()+.1:
                raise ValueError('Reference angles outside observed sweep')
            predicted=np.interp(grid,x[order],y[order]).tolist()
        score=population_band_metrics(predicted,np.atleast_1d(reference['mean']),np.atleast_1d(reference['sd']))
        passed=score['rms_z']<=1 and score['within_2sd_fraction']>=.95
        row.update(status='passed' if passed else 'candidate_failed_gate',predicted=predicted,
                   population=score,reference=reference)
        results.append(row)
    return results


def _array(value, name):
    result = np.asarray(value, dtype=np.float64)
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be nonempty and finite")
    return result


def _paired(predicted, reference):
    predicted, reference = _array(predicted, "predicted"), _array(reference, "reference")
    if predicted.shape != reference.shape:
        raise ValueError("prediction and reference shapes must match; no broadcasting")
    return predicted, reference


def curve_metrics(predicted, reference, *, scale, uncertainty=None):
    """Point-aligned error. Scale is fixed from calibration, never candidate output.

    uncertainty is the reference measurement standard uncertainty (not population
    SD). R2 is undefined for a constant reference and returned as None.
    """
    predicted, reference = _paired(predicted, reference)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("normalization scale must be positive and finite")
    error = predicted - reference
    rmse = float(np.sqrt(np.mean(error**2)))
    variance = float(np.sum((reference - reference.mean())**2))
    result = {
        "n": int(reference.size), "rmse": rmse,
        "mae": float(np.mean(np.abs(error))), "bias": float(error.mean()),
        "max_abs_error": float(np.max(np.abs(error))), "nrmse": rmse / scale,
        "r2": None if variance == 0 else float(1 - np.sum(error**2) / variance),
    }
    if uncertainty is not None:
        uncertainty = _array(uncertainty, "uncertainty")
        if uncertainty.shape != reference.shape or (uncertainty <= 0).any():
            raise ValueError("positive measurement uncertainty required at every reference point")
        result["standardized_rmse"] = float(np.sqrt(np.mean((error / uncertainty)**2)))
        result["within_2u_fraction"] = float(np.mean(np.abs(error) <= 2 * uncertainty))
    return result


def _time(time):
    time = _array(time, "time")
    if time.ndim != 1 or len(time) < 2 or not (np.diff(time) > 0).all():
        raise ValueError("time requires at least two strictly increasing samples")
    return time


def population_band_metrics(predicted, mean, sd):
    """Descriptive population plausibility, NOT uncertainty or a hypothesis test.

    Evaluate only protocol-matched observations (e.g. endpoint ROM or plateau
    strength), never the approach ramp against a maximal-strength distribution.
    SD must describe between-person variation, not SEM, CI or repeated trials.
    Thresholds belong in the predeclared assay contract, not this calculator.
    """
    predicted, mean = _paired(predicted, mean)
    sd = _array(sd, 'population SD')
    if sd.shape != mean.shape or (sd <= 0).any():
        raise ValueError('positive population SD required at every matched point')
    z = (predicted - mean) / sd
    return dict(n=int(z.size), rms_z=float(np.sqrt(np.mean(z*z))),
                mean_z=float(z.mean()), max_abs_z=float(np.abs(z).max()),
                within_1sd_fraction=float(np.mean(np.abs(z) <= 1)),
                within_2sd_fraction=float(np.mean(np.abs(z) <= 2)),
                interpretation='between-person plausibility; not independent biological validation')


def pool_population_summaries(counts, means, sds, *, participant_groups):
    """Combine disjoint, protocol-matched cohorts using sample-variance identity.

    The caller must establish matching units/posture and independent people.
    Repeated sessions, both limbs and crossover arms share a participant group
    and cannot be counted as additional subjects. No interpolation is performed.
    """
    n = _array(counts, 'counts')
    means, sds = _paired(means, sds)
    if (n.ndim != 1 or means.ndim < 1 or len(n) != len(means)
            or (n < 2).any() or (n != np.floor(n)).any() or (sds < 0).any()):
        raise ValueError('integer cohort counts >=2 and nonnegative sample SD required')
    if len(participant_groups) != len(n) or any(not x for x in participant_groups) or len(set(participant_groups)) != len(n):
        raise ValueError('unique nonempty participant groups required; repeated people cannot be pooled')
    weights = n.reshape((-1,) + (1,) * (means.ndim-1))
    total = int(n.sum())
    mean = np.sum(weights * means, axis=0) / total
    variance = np.sum((weights-1)*sds*sds + weights*(means-mean)**2, axis=0)/(total-1)
    return dict(n=total, mean=mean.tolist(), sd=np.sqrt(variance).tolist(),
                participant_groups=list(participant_groups))


def anchored_curve_metrics(predicted, reference, *, anchor_index, scale):
    """One declared calibration sample, all other samples evaluated separately.

    This measures within-curve shape transfer, not independent subject validity.
    No peak normalization, shift, or whole-curve fit is performed.
    """
    predicted,reference = _paired(predicted,reference)
    if predicted.ndim != 1 or len(predicted)<3 or not isinstance(anchor_index,(int,np.integer)) or not 0<=anchor_index<len(predicted):
        raise ValueError('one valid anchor and at least two held-out samples required')
    if predicted[anchor_index]*reference[anchor_index] <= 0:
        raise ValueError('nonzero same-sign anchor torques required')
    factor = float(reference[anchor_index]/predicted[anchor_index])
    heldout = np.arange(len(predicted)) != anchor_index
    return {'anchor_index':int(anchor_index),'amplitude_factor':factor,
            'evaluation_indices':np.flatnonzero(heldout).tolist(),
            'unscaled':curve_metrics(predicted[heldout],reference[heldout],scale=scale),
            'anchored':curve_metrics(factor*predicted[heldout],reference[heldout],scale=scale)}


def _integral(values, time):
    # Explicit trapezoidal integration supports NumPy 1.x and irregular sampling.
    values = np.asarray(values)
    dt = np.diff(time).reshape((-1,) + (1,) * (values.ndim - 1))
    return np.sum(0.5 * (values[1:] + values[:-1]) * dt, axis=0)


def observed_transition_time(time, response, *, initial, target):
    """First 10-to-90% crossing interval; no excitation onset or latency claim.

    Return None if either crossing is absent or the trace starts above 10%.
    This descriptive interval does not identify an activation time constant.
    """
    time = _time(time)
    response = _array(response, 'response')
    if response.shape != time.shape or not np.isfinite([initial,target]).all() or initial == target:
        raise ValueError('paired response and distinct finite reference levels required')
    relative = (response-initial)/(target-initial)
    if relative[0] >= .1:
        return None
    crossings = []
    for level in (.1,.9):
        ids = np.flatnonzero(relative >= level)
        if not len(ids):
            return None
        i = int(ids[0])
        crossings.append(time[i-1]+(level-relative[i-1])*(time[i]-time[i-1])/(relative[i]-relative[i-1]))
    return float(crossings[1]-crossings[0])


def endurance_time_metrics(time, available_capacity, *, target):
    """Sustained-task first capacity < target crossing, with censoring.

    Time starts at task onset (zero). Capacity and the fixed target share units.
    For the 3CC model capacity is resting + active, not active torque: recruitment
    at task onset must not be counted as fatigue failure. A finite observation
    without failure is right-censored, never proof of infinite endurance.
    """
    time = _time(time)
    capacity = _array(available_capacity, 'available_capacity')
    if (capacity.shape != time.shape or time[0] != 0 or (capacity < 0).any()
            or not np.isscalar(target) or not np.isfinite(target) or target <= 0):
        raise ValueError('paired nonnegative capacity, onset zero and positive scalar target required')
    below = np.flatnonzero(capacity < target)
    result = {'endurance_time_s': None, 'observation_duration_s': float(time[-1]),
              'censoring': 'right', 'crossing_bracket_s': None}
    if not len(below):
        return result
    i = int(below[0])
    if i == 0:
        result['censoring'] = 'left'
        return result
    crossing = time[i-1] + (target-capacity[i-1]) * (time[i]-time[i-1]) / (capacity[i]-capacity[i-1])
    result.update(endurance_time_s=float(crossing), censoring='none',
                  crossing_bracket_s=[float(time[i-1]), float(time[i])])
    return result


def recovery_time_metrics(time, capacity, *, initial_capacity, baseline_capacity, fraction=.9):
    """First crossing of a fixed fraction of the externally measured deficit.

    Time zero is recovery onset. Reference initial/baseline levels must be fixed
    before comparing candidates; the observed endpoint is never the baseline.
    Linear interpolation estimates crossing only inside a sampled bracket.
    This is first passage, not proof of sustained recovery or clinical validity.
    """
    time = _time(time)
    capacity = _array(capacity, 'capacity')
    levels = (initial_capacity, baseline_capacity, fraction)
    if (not all(np.isscalar(v) for v in levels) or not np.isfinite(levels).all()
            or not 0 <= initial_capacity < baseline_capacity or not 0 < fraction < 1
            or time[0] != 0 or capacity.shape != time.shape or (capacity < 0).any()):
        raise ValueError('paired nonnegative capacity, onset zero, baseline > initial >= 0 and fraction in (0,1) required')
    threshold = initial_capacity + fraction*(baseline_capacity-initial_capacity)
    result = {'recovery_time_s': None, 'fraction': float(fraction),
              'threshold_capacity': float(threshold), 'censoring': 'right',
              'crossing_bracket_s': None, 'observation_duration_s': float(time[-1])}
    above = np.flatnonzero(capacity >= threshold)
    if not len(above):
        return result
    i = int(above[0])
    if i == 0:
        result['censoring'] = 'left'
        return result
    crossing = time[i-1] + (threshold-capacity[i-1])*(time[i]-time[i-1])/(capacity[i]-capacity[i-1])
    result.update(recovery_time_s=float(crossing), censoring='none',
                  crossing_bracket_s=[float(time[i-1]), float(time[i])])
    return result


def step_response_metrics(time, response, *, initial, target, onset, band=0.02):
    """10-90% transition, overshoot and last-exit settling, without time warping.

    initial/target are externally specified. Incomplete transitions return None.
    The trace must include onset and an initial sample preceding it.
    """
    time = _time(time)
    response = _array(response, "response")
    if response.shape != time.shape or not np.isfinite([initial, target, onset, band]).all():
        raise ValueError("invalid step response parameters")
    if initial == target or not time[0] < onset < time[-1] or not 0 < band < 1:
        raise ValueError("nonzero step, internal onset and 0 < band < 1 required")
    relative = (response - initial) / (target - initial)
    after = time >= onset

    def crossing(level):
        candidates = np.flatnonzero(after & (relative >= level))
        if not len(candidates):
            return None
        i = int(candidates[0])
        if i == 0 or relative[i - 1] >= level:
            return float(time[i] - onset)
        t = time[i - 1] + (level - relative[i - 1]) * (time[i] - time[i - 1]) / (relative[i] - relative[i - 1])
        return float(max(0, t - onset))

    t10, t90 = crossing(0.1), crossing(0.9)
    outside = np.flatnonzero(after & (np.abs(relative - 1) > band))
    settling = (None if len(outside) and outside[-1] == len(time) - 1
                else float(time[outside[-1] + 1] - onset) if len(outside) else 0.0)
    return {
        "t10_s": t10, "t90_s": t90,
        "transition_10_90_s": None if t10 is None or t90 is None else t90 - t10,
        "settling_s": settling,
        "overshoot_fraction": float(max(0, np.max(relative[after]) - 1)),
        "endpoint_error_fraction": float(abs(relative[-1] - 1)),
        "observation_duration_s": float(time[-1] - onset),
    }


def impedance_fit(displacement, velocity, acceleration, applied_torque, *, fixed_inertia=None):
    """Fit external perturbation torque = I*qdd + B*qd + K*q + offset.

    Supply paired perturbation-minus-control signals, NOT the slope of a gait
    torque-angle loop. Columns are scaled before checking identifiability.
    Multi-axis [time, DOF] input identifies full matrices, including coupling;
    matrix row i is output torque i and column j is input motion j. Symmetry
    and passivity are not imposed on measurements that may contain feedback.
    fixed_inertia optionally supplies a separately identified scalar/matrix;
    its uncertainty and provenance must be retained by the owning assay.
    """
    q, torque = _paired(displacement, applied_torque)
    qd, qdd = _array(velocity, "velocity"), _array(acceleration, "acceleration")
    if q.ndim not in (1,2) or qd.shape != q.shape or qdd.shape != q.shape:
        raise ValueError("paired [time] or [time, DOF] perturbation samples required")
    dofs = 1 if q.ndim == 1 else q.shape[1]
    full_design = np.column_stack([qdd, qd, q, np.ones(len(q))])
    design = full_design
    fit_torque = torque
    if fixed_inertia is not None:
        inertia = _array(fixed_inertia, 'fixed_inertia')
        expected = () if q.ndim == 1 else (dofs,dofs)
        if inertia.shape != expected:
            raise ValueError('fixed_inertia must be a scalar or matching DOF matrix')
        matrix = inertia.reshape(1,1) if q.ndim == 1 else inertia
        if not np.allclose(matrix,matrix.T,rtol=0,atol=1e-12) or np.linalg.eigvalsh(matrix).min() < 0:
            raise ValueError('fixed_inertia must be symmetric positive semidefinite')
        fit_torque = torque - (qdd*inertia if q.ndim == 1 else qdd@inertia.T)
        design = full_design[:,dofs:]
    if len(q) < max(8, design.shape[1]):
        raise ValueError("insufficient samples for the full impedance matrix")
    norms = np.linalg.norm(design, axis=0)
    if (norms == 0).any():
        raise ValueError("unexcited impedance parameter")
    scaled = design / norms
    condition = float(np.linalg.cond(scaled))
    if np.linalg.matrix_rank(scaled) != design.shape[1] or condition > 1e6:
        raise ValueError("impedance parameters are not identifiable")
    coefficients = np.linalg.lstsq(scaled, fit_torque, rcond=None)[0]
    coefficients = coefficients / (norms if q.ndim == 1 else norms[:,None])
    if fixed_inertia is not None:
        coefficients = (np.concatenate([inertia.reshape(1),coefficients]) if q.ndim == 1
                        else np.vstack([inertia.T,coefficients]))
    residual = full_design@coefficients-torque
    if q.ndim == 2:
        return {
            'inertia_kg_m2':coefficients[:dofs].T.tolist(),
            'damping_nms_rad':coefficients[dofs:2*dofs].T.tolist(),
            'stiffness_nm_rad':coefficients[2*dofs:3*dofs].T.tolist(),
            'offset_nm':coefficients[-1].tolist(),
            'residual_rmse_nm':np.sqrt(np.mean(residual**2,axis=0)).tolist(),
            'condition_number':condition,'n':len(q),'n_dof':dofs,
            'inertia_mode':'estimated' if fixed_inertia is None else 'fixed',
        }
    return dict(zip(("inertia_kg_m2", "damping_nms_rad", "stiffness_nm_rad", "offset_nm"), map(float, coefficients))) | {
        "residual_rmse_nm": float(np.sqrt(np.mean(residual**2))),
        "condition_number": condition, "n": len(q),
        'inertia_mode':'estimated' if fixed_inertia is None else 'fixed',
    }


def sampling_grid_metrics(time, *, expected_dt, expected_count, terminal_time,
                          tolerance_s=1e-5):
    """Validate pre-step samples on a complete physical-time grid starting at 0.

    A matching count alone cannot detect repeated, missing/replaced, or shifted
    timestamps. terminal_time is the post-step endpoint, not the last sample.
    """
    if (not isinstance(expected_count, int) or isinstance(expected_count, bool)
            or expected_count < 2 or not np.isfinite([expected_dt, terminal_time, tolerance_s]).all()
            or expected_dt <= 0 or tolerance_s < 0):
        raise ValueError('invalid expected sampling grid')
    samples = _array(time, 'time')
    if samples.ndim != 1:
        raise ValueError('time must be one-dimensional')
    count_error = abs(len(samples) - expected_count)
    grid_error = float(np.max(np.abs(samples - np.arange(expected_count)*expected_dt))) if count_error == 0 else None
    endpoint_error = float(abs(terminal_time - expected_count*expected_dt))
    monotonic = bool(len(samples) > 1 and (np.diff(samples) > 0).all())
    return {'count_error': count_error, 'grid_max_error_s': grid_error,
            'terminal_error_s': endpoint_error, 'strictly_increasing': monotonic,
            'tolerance_s': tolerance_s,
            'passed': bool(count_error == 0 and monotonic and grid_error <= tolerance_s and endpoint_error <= tolerance_s)}


def native_clock_grid_metrics(time, *, physics_fps, expected_count, terminal_time):
    """PhysX physics-step callbacks report a float32 step size to Kit's clock.

    Compare to an independently derived grid, never a period fitted to the trace.
    Preserve the requested double-precision grid discrepancy for auditability.
    """
    nominal=1./physics_fps
    native=float(np.float32(nominal))
    result=sampling_grid_metrics(time,expected_dt=native,expected_count=expected_count,terminal_time=terminal_time)
    result.update(nominal_dt_s=nominal,callback_dt_s=native,
        nominal_grid=sampling_grid_metrics(time,expected_dt=nominal,expected_count=expected_count,terminal_time=terminal_time),
        clock_basis='IsaacSim SimulationContext._physics_timer_callback_fn increments current_time by PhysX float32 step_size')
    return result


def torque_trace_metrics(time, *, q, qd, requested, active, elastic, damping,
                         applied, lower, upper, negative_cap, positive_cap,
                         cap_tolerance=1e-5, rom_tolerance_rad=1e-4):
    """Per-axis, time-weighted accounting at the physics substep sampling rate.

    applied is actuator torque sent to the engine. Contact, gravity and joint
    limit reaction forces must be recorded separately and are not included here.
    """
    time = _time(time)
    arrays = {k: _array(v, k) for k, v in locals().copy().items()
              if k in ("q", "qd", "requested", "active", "elastic", "damping", "applied")}
    shape = arrays["q"].shape
    if len(shape) != 2 or shape[0] != len(time) or any(v.shape != shape for v in arrays.values()):
        raise ValueError("all torque traces must have matching [time, dof] shape")
    bounds = {}
    for name, value in (("lower", lower), ("upper", upper), ("negative_cap", negative_cap), ("positive_cap", positive_cap)):
        value = _array(value, name)
        if value.shape not in ((shape[1],), shape):
            raise ValueError(f"invalid {name} shape")
        bounds[name] = np.broadcast_to(value, shape)
    if (bounds["lower"] >= bounds["upper"]).any() or (bounds["negative_cap"] < 0).any() or (bounds["positive_cap"] < 0).any():
        raise ValueError("invalid ROM or strength bounds")
    if not np.isfinite([cap_tolerance, rom_tolerance_rad]).all() or min(cap_tolerance, rom_tolerance_rad) < 0:
        raise ValueError("invalid tolerances")
    duration = time[-1] - time[0]
    req, act = arrays["requested"], arrays["active"]
    saturation = (req > bounds["positive_cap"] + cap_tolerance) | (req < -bounds["negative_cap"] - cap_tolerance)
    violation = np.maximum.reduce([act - bounds["positive_cap"], -act - bounds["negative_cap"], np.zeros(shape)])
    rom_excess = np.maximum.reduce([bounds["lower"] - arrays["q"], arrays["q"] - bounds["upper"], np.zeros(shape)])
    residual = arrays["applied"] - act - arrays["elastic"] - arrays["damping"]
    power = arrays["applied"] * arrays["qd"]
    return {
        "duration_s": float(duration), "n_substeps": len(time),
        "saturation_fraction": (_integral(saturation.astype(float), time) / duration).tolist(),
        "cap_violation_max_nm": violation.max(axis=0).tolist(),
        "rom_excess_max_deg": np.rad2deg(rom_excess.max(axis=0)).tolist(),
        "rom_excess_fraction": (_integral((rom_excess > rom_tolerance_rad).astype(float), time) / duration).tolist(),
        "torque_accounting_max_nm": np.abs(residual).max(axis=0).tolist(),
        "positive_work_j": _integral(np.maximum(power, 0), time).tolist(),
        "negative_work_j": _integral(np.minimum(power, 0), time).tolist(),
        "damping_energy_injection_j": _integral(np.maximum(arrays["damping"] * arrays["qd"], 0), time).tolist(),
    }


def repeatability_limits(mean, sd, icc):
    """Reliability-derived SEM and paired-change MDC95, not model accuracy.

    The population SD alone is not measurement uncertainty. The ICC and SD
    must describe the same measurement protocol and reliability population.
    """
    if not np.isfinite([mean,sd,icc]).all() or mean == 0 or sd < 0 or not 0 <= icc <= 1:
        raise ValueError('finite nonzero mean, nonnegative SD and ICC in [0,1] required')
    sem = float(sd*np.sqrt(1-icc))
    mdc = float(1.96*np.sqrt(2)*sem)
    return {'sem':sem, 'sem_percent_of_mean':100*sem/abs(mean),
            'mdc95':mdc, 'mdc95_percent_of_mean':100*mdc/abs(mean),
            'interpretation':'Repeatability of matched measurements; not a biological model acceptance threshold'}


def contract_gate(value, metric_id, *, evidence_kind, required_evidence=None, contract_path=None):
    """Use the declared threshold while preserving the assay's evidence level.

    An explicit evidence override is a screening/source-equation comparison,
    not satisfaction of the contract's independent measurement requirement.
    """
    path = Path(contract_path) if contract_path is not None else v1_resource('validation_contract.json')
    raw = path.read_bytes()
    rows = [r for r in json.loads(raw)['metrics'] if r['id'] == metric_id]
    if len(rows) != 1:
        raise ValueError(f'Expected exactly one contract metric: {metric_id}')
    row = rows[0]
    required = row['required_evidence'] if required_evidence is None else required_evidence
    result = gate(value, row['threshold_max'], evidence_kind=evidence_kind, required_evidence=required)
    result.update(metric_id=metric_id, threshold_basis=row['threshold_basis'],
                  contract_sha256=hashlib.sha256(raw).hexdigest(),
                  contract_required_evidence=row['required_evidence'],
                  assay_required_evidence=required,
                  contract_evidence_satisfied=evidence_kind == row['required_evidence'])
    return result


def gate(value, threshold, *, evidence_kind, required_evidence="independent_measurement"):
    """Explicit epistemic gate. Missing or inapplicable evidence never passes."""
    if value is None or threshold is None:
        return {"status": "not_evaluated", "value": value, "threshold": threshold}
    if not np.isfinite([value, threshold]).all() or threshold < 0:
        raise ValueError("gate values and nonnegative threshold must be finite")
    if evidence_kind != required_evidence:
        return {"status": "insufficient_evidence", "value": float(value), "threshold": float(threshold),
                "evidence_kind": evidence_kind, "required_evidence": required_evidence}
    return {"status": "pass" if value <= threshold else "fail", "value": float(value), "threshold": float(threshold),
            "evidence_kind": evidence_kind}


def shared_state_error_bound(states_a, reference_a, states_b, reference_b, *, scale):
    """Lower bound on max curve NRMSE for any identical-state deterministic model.

    Exact shared states only; no rounding or interpolation. The triangle
    inequality on shared samples gives ||ya-yb|| / (scale*(sqrt(na)+sqrt(nb))).
    Full curve sample counts remain in the denominator, as in curve_metrics.
    Different subjects/protocols may explain the conflict; this does not reject
    either measurement or establish a biological noise floor.
    """
    a,b = _array(states_a,'states_a'),_array(states_b,'states_b')
    ya,yb = _array(reference_a,'reference_a'),_array(reference_b,'reference_b')
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or ya.shape != (len(a),) or yb.shape != (len(b),):
        raise ValueError('state matrices and scalar reference curves required')
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('positive finite shared normalization required')
    if len(np.unique(a,axis=0)) != len(a) or len(np.unique(b,axis=0)) != len(b):
        raise ValueError('duplicate states require explicit trial aggregation')
    ia,ib = np.nonzero((a[:,None,:] == b[None,:,:]).all(-1))
    bound = float(np.linalg.norm(ya[ia]-yb[ib])/(scale*(np.sqrt(len(a))+np.sqrt(len(b)))))
    return {'shared_samples':len(ia),'sample_indices_a':ia.tolist(),'sample_indices_b':ib.tolist(),
            'minimum_possible_max_curve_nrmse':bound,
            'condition':'same deterministic state-to-torque mapping and fixed shared normalization'}


def rom_boundary_metrics(predicted_boundary, reference_boundary, predicted_allowed, reference_allowed):
    """Boundary angles in radians, classifications from a fixed boundary-pose grid."""
    p, r = _paired(predicted_boundary, reference_boundary)
    pa, ra = np.asarray(predicted_allowed), np.asarray(reference_allowed)
    if pa.dtype != bool or ra.dtype != bool or pa.size == 0 or pa.shape != ra.shape:
        raise ValueError("paired nonempty boolean ROM classification arrays required")
    return {"rom_boundary_error_deg": float(np.rad2deg(np.max(np.abs(p-r)))),
            "rom_false_accept_fraction": float(np.mean(pa[~ra])) if (~ra).any() else None,
            "rom_false_reject_fraction": float(np.mean(~pa[ra])) if ra.any() else None,
            "n_allowed": int(ra.sum()), "n_forbidden": int((~ra).sum())}


def contact_metrics(force, reference_force, cop, reference_cop, *, mass_kg, min_force_n=20):
    """Force [time,xyz] and CoP [time,xy], common world-aligned plate frame.

    CoP may be NaN during flight; stance missingness is counted, not hidden.
    Net body/foot force cannot substitute for unmeasured plantar pressure.
    """
    force, reference_force = _paired(force, reference_force)
    cop, reference_cop = np.asarray(cop, dtype=float), np.asarray(reference_cop, dtype=float)
    if force.ndim != 2 or force.shape[1] != 3 or cop.shape != (len(force), 2) or reference_cop.shape != cop.shape:
        raise ValueError("expected aligned force[time,3], cop[time,2]")
    if not np.isfinite([mass_kg, min_force_n]).all() or mass_kg <= 0 or min_force_n <= 0:
        raise ValueError("positive mass and force threshold required")
    stance = (force[:, 2] >= min_force_n) & (reference_force[:, 2] >= min_force_n)
    valid = stance & np.isfinite(cop).all(1) & np.isfinite(reference_cop).all(1)
    coverage = float(valid.sum()/stance.sum()) if stance.any() else 0.
    error = cop[valid] - reference_cop[valid]
    return {"grf_rmse_bw": (np.sqrt(np.mean((force-reference_force)**2, axis=0))/(mass_kg*9.81)).tolist(),
            "cop_rmse_m": float(np.sqrt(np.mean(np.sum(error**2,axis=1)))) if valid.any() else None,
            "cop_coverage_fraction": coverage, "paired_stance_samples": int(stance.sum()),
            "stance_mismatch_fraction": float(np.mean((force[:,2]>=min_force_n)!=(reference_force[:,2]>=min_force_n)))}


def residual_decomposition(predicted, reference, *, scale):
    """Describe offset versus varying error; centering is NOT a corrected gate.

    Bias uses the entire evaluated curve, so this diagnostic cannot establish
    held-out accuracy or identify a physiological/measurement offset.
    """
    original = curve_metrics(predicted, reference, scale=scale)
    predicted, reference = _paired(predicted, reference)
    if predicted.ndim != 1 or len(predicted) < 3:
        raise ValueError("at least three paired one-dimensional samples required")
    error = predicted-reference
    bias = float(error.mean())
    mse = float(np.mean(error**2))
    centered_mse = float(np.mean((error-bias)**2))
    return {"absolute": original, "bias_nm": bias,
            "centered_rmse_nm": float(np.sqrt(centered_mse)),
            "centered_nrmse": float(np.sqrt(centered_mse)/scale),
            "bias_fraction_of_mse": bias*bias/mse if mse else None,
            "mse_identity_error_nm2": abs(mse-bias*bias-centered_mse),
            "interpretation": "Descriptive decomposition only; no corrected biological gate"}


def cycle_work(angle, external_torque, *, closure_tolerance_rad=1e-5):
    """External work integral over one closed cycle, joules; clockwise sign retained."""
    angle, external_torque = _paired(angle, external_torque)
    if angle.ndim != 1 or len(angle) < 3 or abs(angle[-1]-angle[0]) > closure_tolerance_rad:
        raise ValueError("a complete closed one-dimensional angle cycle is required")
    return float(np.sum(.5*(external_torque[1:]+external_torque[:-1])*np.diff(angle)))


def active_rom_metrics(trace, substeps, audit, chapters):
    """Selected active reachability; mechanics only, no population AROM claim.

    Torques/velocities use pre-step 480 Hz samples; held-angle and non-test
    tracking use rendered-grid post-step samples. Test-limb contact is 30 Hz.
    """
    names=list(trace['dof_names']);body_names=list(trace['body_names'])
    time=np.asarray(substeps['time']);audit_time=np.asarray(audit['time'])
    if not np.array_equal(time,audit_time):
        raise ValueError('Active ROM torque audit must cover every physical substep')
    limbs={'Hip':['Hip','Knee','Ankle','Toe'],'Knee':['Knee','Ankle','Toe'],
           'Ankle':['Ankle','Toe'],'Toe':['Toe'],'Elbow':['Elbow','Wrist','Hand'],
           'Wrist':['Wrist','Hand']}
    rows=[]
    for ci,c in enumerate(chapters):
        j=names.index(c['joint']);sm=np.asarray(audit['chapter_index'])==ci
        fm=np.asarray(trace['chapter_index'])==ci
        if not sm.any() or not fm.any() or np.any(np.asarray(audit['focus'])[sm]!=j):
            raise ValueError('Missing or mismatched active ROM chapter audit')
        local=time[sm]-c['start_s'];frame_local=np.asarray(trace['time'])[fm]-c['start_s']
        hold=frame_local>c['duration_s']-2+1e-6
        rest=local>=c['duration_s']-1
        if not hold.any() or not rest.any():raise ValueError('Missing endpoint hold')
        q=np.asarray(substeps['q'])[sm,j];qd=np.asarray(substeps['qd'])[sm,j]
        active=np.asarray(substeps['active'])[sm,j];requested=np.asarray(substeps['requested'])[sm,j]
        other=np.arange(len(names))!=j
        non_test=np.rad2deg(np.abs(np.asarray(trace['q'])[fm][:,other]-np.asarray(trace['target'])[fm][:,other]))
        side,body,_=c['joint'].split('_')
        indices=[body_names.index(side+'_'+b) for b in limbs[body]]
        contacts=np.linalg.norm(np.asarray(trace['contact_body_forces_w'])[fm][:,indices,:],axis=-1)
        error=np.rad2deg(q)-np.rad2deg(np.asarray(audit['target_rad'])[sm])
        final_q=np.rad2deg(np.asarray(trace['q'])[fm,j][hold])
        step_dt=float(np.median(np.diff(time)))
        all_q=np.asarray(substeps['q'])[sm]
        all_active=np.asarray(substeps['active'])[sm]
        rom_excess=np.maximum(np.asarray(trace['lower'])-all_q,all_q-np.asarray(trace['upper'])).clip(min=0)
        cap_excess=np.maximum(all_active-np.asarray(substeps['positive_cap'])[sm],
            -all_active-np.asarray(substeps['negative_cap'])[sm]).clip(min=0)
        closure=np.asarray(substeps['applied'])[sm]-all_active-np.asarray(substeps['elastic'])[sm]-np.asarray(substeps['damping'])[sm]
        rows.append(dict(label=c['label'],joint=c['joint'],start_deg=c['start_deg'],target_deg=c['end_deg'],
            endpoint_mean_deg=float(final_q.mean()),endpoint_error_max_deg=float(np.abs(final_q-c['end_deg']).max()),
            tracking_max_deg=float(np.abs(error).max()),peak_speed_deg_s=float(np.rad2deg(np.abs(qd)).max()),
            last_second_peak_to_peak_deg=float(np.rad2deg(np.ptp(q[rest]))),
            non_test_tracking_max_deg=float(non_test.max()),test_limb_contact_max_n=float(contacts.max()),
            non_test_active_max_nm=float(np.abs(all_active[:,other]).max()),
            test_axis_fixture_max_nm=float(np.abs(np.asarray(audit['fixture_focus_nm'])[sm]).max()),
            active_min_nm=float(active.min()),active_max_nm=float(active.max()),
            saturation_fraction=float((np.abs(requested-active)>1e-4).mean()),
            all_joint_peak_speed_deg_s=float(np.rad2deg(np.abs(np.asarray(substeps['qd'])[sm])).max()),
            rom_excess_max_deg=float(np.rad2deg(rom_excess.max())),cap_violation_max_nm=float(cap_excess.max()),
            torque_accounting_max_nm=float(np.abs(closure).max()),
            active_work_j=float(np.sum(active*qd)*step_dt),physical_samples=int(sm.sum())))
    return rows


def all_direction_rom_metrics(trace, substeps, audit, chapters, thresholds):
    """Endpoint, boundary and recovery windows; preserve failures and lock scope."""
    names=list(trace['dof_names']);rows=[]
    for ci,c in enumerate(chapters):
        sm=np.asarray(audit['chapter_index'])==ci;fm=np.asarray(trace['chapter_index'])==ci
        j=names.index(c['joint']);other=np.arange(len(names))!=j
        t=np.asarray(trace['time'])[fm]-c['start_s'];q=np.rad2deg(np.asarray(trace['q'])[fm])
        target=np.rad2deg(np.asarray(trace['target'])[fm]);active=np.asarray(substeps['active'])[sm]
        request=np.asarray(substeps['requested'])[sm];all_q=np.asarray(substeps['q'])[sm]
        def window(key):
            a,b=c[key];mask=(t>a+1e-5)&(t<=b+1e-5)
            if not mask.any():raise ValueError(f'Missing {key}: {c["label"]}')
            return mask
        endpoint=window('endpoint_window_s');recovery=window('recovery_window_s');probe=window('probe_window_s')
        if c['locked_axis']:
            # Exclude the explicitly ramped external probe release from the
            # boundary-reaction plateau, without excluding it from speed gates.
            probe &= t<=c['probe_window_s'][1]-.25+1e-5
            if not probe.any():raise ValueError('Missing constant external-probe plateau')
        sub_local=np.asarray(audit['time'])[sm]-c['start_s'] if 'time' in audit else t
        a,b=c['endpoint_window_s'];sub_hold=(sub_local>a+1e-5)&(sub_local<=b+1e-5)
        if not sub_hold.any():raise ValueError('Missing high-rate hold samples')
        bounds=np.rad2deg(np.stack([trace['lower'],trace['upper']]))
        excess=np.maximum(bounds[0]-q,q-bounds[1]).clip(min=0)
        sub_excess=np.maximum(np.asarray(trace['lower'])-all_q,all_q-np.asarray(trace['upper'])).clip(min=0)
        cap=np.maximum(active-np.asarray(substeps['positive_cap'])[sm],-active-np.asarray(substeps['negative_cap'])[sm]).clip(min=0)
        closure=np.asarray(substeps['applied'])[sm]-active-np.asarray(substeps['elastic'])[sm]-np.asarray(substeps['damping'])[sm]
        locked=c['locked_axis'];allowed=0.05 if locked else thresholds['endpoint_error_deg']
        row=dict(joint=c['joint'],direction=c['direction'],locked_axis=locked,label=c['label'],
            endpoint_mean_deg=float(q[endpoint,j].mean()),endpoint_error_deg=float(np.abs(q[endpoint,j]-c['end_deg']).max()),
            recovery_error_deg=float(np.abs(q[recovery,j]-c['start_deg']).max()),
            hold_peak_to_peak_deg=float(np.ptp(q[endpoint,j])),
            rom_excess_deg=float(max(excess.max(),np.rad2deg(sub_excess.max()))),
            non_test_error_deg=float(np.abs(q[:,other]-target[:,other]).max()),
            cap_error_nm=float(cap.max()),torque_sum_error_nm=float(np.abs(closure).max()),
            active_max_abs_nm=float(np.abs(active[:,j]).max()),non_test_active_nm=float(np.abs(active[:,other]).max()),
            fixture_focus_max_nm=float(np.abs(np.asarray(audit['fixture_focus_nm'])[sm]).max()),
            saturation_fraction=float((np.abs(request[:,j]-active[:,j])>1e-4).mean()),
            peak_speed_deg_s=float(np.rad2deg(np.abs(np.asarray(substeps['qd'])[sm,j])).max()),
            probe_angle_mean_deg=float(q[probe,j].mean()),physical_samples=int(sm.sum()),
            contact_points_max=int(np.asarray(trace['contact_points_count'])[fm].max()),
            contact_penetration_m=float(max(0,-np.asarray(trace['contact_min_separation_m'])[fm].min())))
        row['hold_active_peak_to_peak_nm']=float(np.ptp(active[sub_hold,j]))
        row['hold_physics_peak_to_peak_deg']=float(np.rad2deg(np.ptp(all_q[sub_hold,j])))
        row['hold_physics_peak_speed_deg_s']=float(np.rad2deg(np.abs(np.asarray(substeps['qd'])[sm,j][sub_hold]).max()))
        capacity=max(float(np.asarray(substeps['positive_cap'])[sm,j].max()),float(np.asarray(substeps['negative_cap'])[sm,j].max()))
        row['hold_torque_ripple_threshold_nm']=max(.5,.05*capacity)
        power=active[:,j]*np.asarray(substeps['qd'])[sm,j]
        dt=float(np.median(np.diff(sub_local)))
        outbound=(sub_local>=1)&(sub_local<=c['endpoint_window_s'][0]-.5)
        row['active_positive_work_j']=float(np.maximum(power,0).sum()*dt)
        row['active_negative_work_j']=float(np.minimum(power,0).sum()*dt)
        row['outbound_active_work_j']=float(power[outbound].sum()*dt)
        bound=c['rom_deg'][1 if c['direction']>0 else 0]
        row['probe_boundary_distance_deg']=float(np.abs(q[probe,j]-bound).mean())
        residual=np.asarray(trace['inertia_accel'])[fm,j]+np.asarray(trace['coriolis'])[fm,j]+np.asarray(trace['gravity'])[fm,j]-np.asarray(trace['engine_total'])[fm,j]
        row['probe_generalized_residual_mean_nm']=float(residual[probe].mean())
        row['probe_residual_opposes_input']=row['probe_generalized_residual_mean_nm']*c['direction']<-.001
        row['limit_test_status']='exercised' if row['probe_boundary_distance_deg']<=.01 and row['probe_residual_opposes_input'] else 'not_exercised'
        row['residual_scope']='Unresolved constraint/contact/discretization contribution; not an isolated reaction measurement'
        gates={k:row[k]<=thresholds[k] for k in ('rom_excess_deg','non_test_error_deg','cap_error_nm','torque_sum_error_nm')}
        gates.update(finite=bool(np.isfinite(q).all() and np.isfinite(active).all()),
            no_hidden_active=row['non_test_active_nm']<=2e-4,contact_penetration=row['contact_penetration_m']<=.005,
            speed=row['peak_speed_deg_s']<=60,
            physical_hold_stability=row['hold_physics_peak_to_peak_deg']<=1 and row['hold_physics_peak_speed_deg_s']<=5,
            hold_torque_ripple=row['hold_active_peak_to_peak_nm']<=row['hold_torque_ripple_threshold_nm'])
        if locked:
            gates.update(active_blocked=row['active_max_abs_nm']<=2e-4,
                narrow_rom=row['rom_excess_deg']<=allowed,
                external_probe_present=abs(row['fixture_focus_max_nm']-c['external_probe_nm'])<=1e-5,
                boundary_exercised=row['limit_test_status']=='exercised')
        else:
            gates.update(endpoint=row['endpoint_error_deg']<=allowed,
                recovery=row['recovery_error_deg']<=thresholds['recovery_error_deg'],
                stable_hold=row['hold_peak_to_peak_deg']<=thresholds['hold_peak_to_peak_deg'],
                no_external_assistance=row['fixture_focus_max_nm']<=1e-8,
                boundary_exercised=row['limit_test_status']=='exercised')
        row.update(checks=gates,status='passed' if all(gates.values()) else 'candidate_failed_gate')
        rows.append(row)
    return rows


def frequency_response_metrics(predicted, reference):
    """Complex transfer gains on the SAME externally specified frequency grid."""
    predicted, reference = np.asarray(predicted, complex), np.asarray(reference, complex)
    if not predicted.size or predicted.shape != reference.shape or not np.isfinite(predicted).all() or not np.isfinite(reference).all():
        raise ValueError("paired finite complex frequency responses required")
    if (np.abs(reference) == 0).any() or (np.abs(predicted) == 0).any():
        raise ValueError("zero response has undefined gain ratio/phase")
    ratio = predicted/reference
    return {"frequency_gain_error_db": float(np.max(np.abs(20*np.log10(np.abs(ratio))))),
            "frequency_phase_error_deg": float(np.max(np.abs(np.rad2deg(np.angle(ratio)))))}
