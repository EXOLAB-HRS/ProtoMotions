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
"""Reproducible quantitative assays for the existing healthy_adult_v1 candidate.

Run from ProtoMotions: python -m protomotions.robot_configs.human_model.validation
    --output ../output/... --stage screening --seed 2100 --features strength,activation
No tuning occurs here. A source-equation audit is never a biological gate pass.
"""

import argparse
import contextlib
import csv
import hashlib
import json
import math
from pathlib import Path
from .paths import v1_resource, PACKAGE_ROOT, PROTOMOTIONS_ROOT, WORKSPACE_ROOT, ASSET_ROOT
import subprocess
import time

import numpy as np
import torch

from .dynamics import HumanJointModel
from .metrics import curve_metrics, gate, step_response_metrics, torque_trace_metrics, sampling_grid_metrics
from .profile import load_profile


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluation_requirements(contract):
    """Separate plant requirements from optional integration, without passing either.

    These are unevaluated requirements, not aggregated assay outcomes. Historical
    run gates retain their original scope and thresholds in the evidence index.
    """
    supplementary = contract['scope_policy']['supplementary_groups']
    coverage = contract['implementation_coverage']
    if set(coverage) != {row['group'] for row in contract['metrics']}:
        raise ValueError('Every metric group needs an explicit implementation coverage entry')
    result = {'joint_level_plant': {}, 'supplementary_integration': {}}
    for row in contract['metrics']:
        scope = supplementary.get(row['group'], 'joint_level_plant')
        bucket = 'joint_level_plant' if scope == 'joint_level_plant' else 'supplementary_integration'
        result[bucket][row['id']] = {
            **gate(None, row['threshold_max'], evidence_kind=row['required_evidence'],
                   required_evidence=row['required_evidence']),
            'scope': scope, 'group': row['group'], 'unit': row['unit'],
            'threshold_basis': row['threshold_basis'],
            'required_evidence': row['required_evidence'],
            'conditions': row['conditions'],
            'implementation':dict(coverage[row['group']]),
        }
    return result


def _source_strength(row, cohorts, angle, velocity, weights=(.5,.5)):
    """Scalar source Eq.9; independent of tensor implementation and its guards."""
    result = 0.0
    for (c1, c2, c3, c4, c5, c6), cohort, weight in zip(row["coefficients"], cohorts, weights):
        c1 *= cohort["mass_kg"]*9.81*cohort["height_m"]
        v = abs(velocity)
        r = (2*c4*c5+v*(c5-3*c4))/(2*c4*c5+v*(2*c5-4*c4))
        result += weight*c1*max(0, math.cos(c2*(angle-c3)))*max(0, r)*(1-c6*min(velocity, 0))
    return result


def resource_snapshot():
    result = {}
    for name, command in {
        "ram": ["free", "-h"], "swap": ["swapon", "--show"],
        "gpu": ["nvidia-smi"],
        "processes": ["ps", "-u", str(__import__('os').getuid()), "-o", "pid,ppid,etime,rss,pcpu,args", "--sort=-rss"],
    }.items():
        try:
            p = subprocess.run(command, text=True, capture_output=True, timeout=10)
            result[name] = {"exit_code": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
        except (OSError, subprocess.TimeoutExpired) as e:
            result[name] = {"error": str(e)}
    return result


def numerical_assays(model, seed):
    generator = torch.Generator().manual_seed(seed)
    q = model.lower + torch.rand(128, len(model.names), generator=generator, dtype=torch.float64)*(model.upper-model.lower)
    # In-ROM includes challenging coupled extreme configurations; external
    # extrapolation is a separate numerical audit, not a healthy-pose sample.
    qd = 10*torch.randn(q.shape, generator=generator, dtype=q.dtype)
    result = {"energy": {}}
    for scope, positions in (("in_rom", q), ("extrapolation", q*3)):
        force = model.elastic_torque(positions)
        h = 1e-6
        gradient = torch.empty_like(positions)
        for i in range(len(model.names)):
            step = torch.zeros_like(positions); step[:, i] = h
            gradient[:, i] = (model.potential_energy(positions+step)-model.potential_energy(positions-step))/(2*h)
        error = float(torch.sqrt(torch.mean((force+gradient)**2))/torch.sqrt(torch.mean(force**2)).clamp_min(1))
        result["energy"][scope] = {"nrmse": error, "gate": gate(error, 1e-6, evidence_kind="numerical", required_evidence="numerical")}
        flags=model.passive_guard_flags(positions,qd)
        result['energy'][scope]['passive_guard_diagnostic']={
            'scope':'Random independent ROM-axis box samples, not a healthy-pose distribution',
            'component_names':model.passive_component_names,
            'dof_names':model.names,
            'fraction_by_component_or_dof':{key:value.double().mean(0).tolist() for key,value in flags.items()},
            'any_guard_sample_fraction':float(torch.cat(list(flags.values()),dim=-1).any(-1).double().mean()),
        }
    injection = float((model.damping_torque(qd)*qd).clamp_min(0).max())
    result["damping"] = gate(injection, 1e-10, evidence_kind="numerical", required_evidence="numerical")
    negative, positive, outside = model.strength_caps(q, qd)
    result["strength_outside_domain_fraction_random_rom_diagnostic"] = outside.double().mean(0).tolist()
    requested = 1000*torch.randn(q.shape, generator=generator, dtype=q.dtype)
    total, active, passive = model.torques(requested, q, qd, dt=.001)
    excess = float(torch.maximum(active-positive, -active-negative).clamp_min(0).max())
    result["cap"] = gate(excess, 2e-4, evidence_kind="numerical", required_evidence="numerical")
    result["backend_ceiling_excess_nm"] = float((total.abs()-model.backend_limit).clamp_min(0).max())
    result["torque_accounting"] = gate(float((total-active-passive).abs().max()), 2e-4, evidence_kind="numerical", required_evidence="numerical")
    result['passive_impedance'] = passive_impedance_assay(model, seed)
    return result


def passive_impedance_assay(model, seed):
    """Audit local tissue tangents at fixed random poses, without a controller."""
    generator = torch.Generator().manual_seed(seed)
    q = model.lower + (.1 + .8*torch.rand(12, len(model.names), generator=generator,
                                       dtype=model.lower.dtype))*(model.upper-model.lower)
    qd = 30*torch.randn(q.shape, generator=generator, dtype=q.dtype)
    result = {'evidence_kind':'numerical', 'scope':'Passive tissue only; not biological impedance validation',
              'seed':seed, 'dof_names':model.names, 'conditions':{}}
    for label, positions in (('in_rom',q), ('extrapolation',3*q)):
        tangent = model.passive_impedance(positions,qd)
        h = 1e-6
        reference = torch.empty((*positions.shape,len(model.names)),dtype=q.dtype)
        for j in range(len(model.names)):
            step = torch.zeros_like(positions); step[:,j] = h
            reference[:,:,j] = -(model.elastic_torque(positions+step)-model.elastic_torque(positions-step))/(2*h)
        stiffness = tangent['stiffness_nm_per_rad']
        damping = -(model.damping_torque(qd+h)-model.damping_torque(qd-h))/(2*h)
        result['conditions'][label] = {
            'q_rad':positions.tolist(), 'qd_rad_s':qd.tolist(),
            'differentiable':tangent['differentiable'].tolist(),
            'stiffness_diagonal_nm_per_rad':stiffness.diagonal(dim1=-2,dim2=-1).tolist(),
            'damping_nms_per_rad':tangent['damping_nms_per_rad'].tolist(),
            'stiffness_derivative_max_error_nm_per_rad':float((stiffness-reference).abs().max()),
            'damping_derivative_max_error_nms_per_rad':float((tangent['damping_nms_per_rad']-damping).abs().max()),
            'reciprocity_max_error_nm_per_rad':float((stiffness-stiffness.transpose(-1,-2)).abs().max()),
            'minimum_stiffness_eigenvalue_nm_per_rad':float(torch.linalg.eigvalsh(stiffness).min()),
        }
    return result


def wrist_damping_assay(model):
    """Reproduce the reported Hz-regression definition, not measured waveforms."""
    params = json.loads(v1_resource('candidate_parameters.json').read_text())['wrist_damping']
    phase = torch.linspace(0,2*math.pi,1025,dtype=model.lower.dtype)[:-1]
    amplitude = math.radians(params['oscillation_amplitude_deg'])
    frequencies = np.arange(params['frequency_hz'][0],params['frequency_hz'][1]+1)
    target = .5*(params['reported_vi_female']+params['reported_vi_male'])
    result = {'scope':'Source-definition audit only; constant damping cannot reproduce nonlinear Figure2 curves',
              'source':params['source'], 'frequency_hz':frequencies.tolist(),
              'reference_slope_per_hz':target,'wrists':{}}
    for name in params['joint_names']:
        i = model.names.index(name)
        quadrature = []
        for f in frequencies:
            qd = torch.zeros((len(phase),len(model.names)),dtype=phase.dtype)
            qd[:,i] = amplitude*2*math.pi*f*torch.cos(phase)
            torque = model.damping_torque(qd)[:,i]
            quadrature.append(float(-2*(torque*torch.cos(phase)).mean()/amplitude))
        slope, intercept = np.polyfit(frequencies,quadrature,1)
        result['wrists'][name] = {
            'damping_nms_per_rad':float(model.damping[i]),
            'quadrature_stiffness_nm_per_rad':quadrature,
            'slope_per_hz':float(slope),'intercept_nm_per_rad':float(intercept),
            'definition_gate':gate(abs(float(slope)-target),1e-8,evidence_kind='source_equation',required_evidence='source_equation'),
            'biological_gate':gate(None,.2,evidence_kind='independent_measurement'),
        }
    return result


def strength_assays(model):
    params = json.loads(v1_resource("candidate_parameters.json").read_text())["strength"]
    if model.strength_reference_size is not None:
        mass,height = model.strength_reference_size
        params['cohorts'] = [dict(c,mass_kg=mass,height_m=height) for c in params['cohorts']]
    results, samples = {}, []
    for direction, row in params["directions"].items():
        predicted, reference = [], []
        # Grid is a source equation audit; no independent human measurements.
        angles = np.linspace(*row["angle_domain_rad"], 17)
        velocities = np.linspace(-row["eccentric_limit_rad_s"], row["concentric_limit_rad_s"], 13)
        for side in ("L", "R"):
            i = model.names.index(f"{side}_{row['joint']}")
            for angle in angles:
                for velocity in velocities:
                    q = torch.zeros(1, len(model.names), dtype=torch.float64); qd = q.clone()
                    q[0, i] = angle*row["clinical_angle_sign"]; qd[0, i] = velocity*row["sign"]
                    if row['joint']=='Ankle_y':
                        # Anderson ankle testing used knee flexion 50 degrees.
                        q[0,model.names.index(f'{side}_Knee_y')]=math.radians(50)
                    neg, pos, _ = model.strength_caps(q, qd)
                    pred = float((pos if row["sign"] > 0 else neg)[0, i])
                    ref = _source_strength(row, params["cohorts"], angle, velocity,model.strength_cohort_weights.tolist())
                    ref *= float(model.active_strength_scale[i,int(row['sign']>0)])
                    predicted.append(pred); reference.append(ref)
                    samples.append((side,direction,angle,velocity,pred,ref))
        # A deliberately disabled direction has zero reference amplitude; use
        # 1 Nm solely for transcription accounting, never biological NRMSE.
        errors = curve_metrics(predicted, reference, scale=max(1.,max(reference)))
        results[direction] = {"source_equation_error": errors,
            "source_gate": gate(errors["max_abs_error"], 1e-8, evidence_kind="source_equation", required_evidence="source_equation"),
            "biological_gate": gate(errors["nrmse"], .15, evidence_kind="source_equation")}
    return results, samples


def activation_assays(model):
    if "activation" not in model.features:
        return {"status": "not_enabled"}
    # Use a fresh batch after the vectorized numerical audit.
    model._activation = None
    q = torch.zeros(1, len(model.names), dtype=torch.float64)
    i = model.names.index("L_Knee_y"); q[0,i] = 1.2
    neg, pos, _ = model.strength_caps(q, torch.zeros_like(q))
    request = torch.zeros_like(q); request[0,i] = pos[0,i]*.5
    dt = .0005
    clock = np.arange(-.01,.5+dt/2,dt)
    response = []
    for t in clock:
        _, active, _ = model.torques(request if t >= 0 else request*0, q, q*0, dt=dt)
        response.append(float(active[0,i]))
    # The return value is the end-of-step activation; align its sample time.
    measured = step_response_metrics(clock+dt, response, initial=0, target=float(request[0,i]), onset=0)
    expected = model.activation_rise_s*math.log(9)
    error = abs(measured["transition_10_90_s"]-expected)
    return {"rise": measured, "analytic_10_90_s": expected,
            "numerical_time_error_gate": gate(error, dt, evidence_kind="numerical", required_evidence="numerical"),
            "biological_gate": gate(None,.01,evidence_kind="independent_measurement")}


def mujoco_assay(features, trace_path, steps=30):
    """Actual backend, free root, zero gravity/no contact: actuator accounting only."""
    import os
    import mujoco
    from protomotions.robot_configs.smpl import SmplRobotConfig
    from protomotions.simulator.mujoco.config import MujocoSimulatorConfig
    from protomotions.simulator.mujoco.simulator import MujocoSimulator
    from protomotions.components.scene_lib import SceneLib, SceneLibConfig
    os.environ["PROTOMOTIONS_HUMAN_MODEL_FEATURES"] = ",".join(features)
    sim = MujocoSimulator(MujocoSimulatorConfig(headless=True,num_envs=1,experiment_name="human_model_validation"),
        SmplRobotConfig(), None, torch.device("cpu"), SceneLib(SceneLibConfig()))
    sim._initialize_with_markers(None)
    sim.model.opt.gravity[:] = 0; sim.data.qpos[2] = 3
    sim.model.geom_contype[:] = 0; sim.model.geom_conaffinity[:] = 0
    sim._common_actions.zero_()
    initial_wrist_velocity_rad_s = {'L_Wrist_x':1.0,'R_Wrist_x':-1.0}
    sim_order = sim.data_conversion.dof_convert_to_sim.cpu().tolist()
    for name, velocity in initial_wrist_velocity_rad_s.items():
        common_index = sim._human_joint_model.names.index(name)
        sim.data.qvel[6+sim_order.index(common_index)] = velocity
    mujoco.mj_forward(sim.model,sim.data)
    data = {name: [] for name in ("time", "q", "qd", "requested", "active", "elastic", "damping", "applied", "negative_cap", "positive_cap")}
    def record(s,state):
        data["time"].append(s.data.time)
        data["q"].append(state.dof_pos[0].detach().cpu().numpy().copy())
        data["qd"].append(state.dof_vel[0].detach().cpu().numpy().copy())
        for key, attr in (("requested","human_requested_torques"),("active","human_active_torques"),
                          ("elastic","human_elastic_torques"),("damping","human_damping_torques"),
                          ("applied","human_applied_torques"),("negative_cap","human_negative_caps"),("positive_cap","human_positive_caps")):
            data[key].append(getattr(s,attr)[0].detach().cpu().numpy().copy())
    sim.human_model_trace_callback = record
    engine_samples = []
    for _ in range(steps):
        previous_time = float(sim.data.time)
        sim._physics_step()
        expected = sim.human_applied_torques[0,sim.data_conversion.dof_convert_to_sim].cpu().numpy()
        # Read engine generalized actuator force, independently of the trace
        # callback's requested/applied tensors. This samples each control end.
        actual = sim.data.qfrc_actuator[6:6+len(expected)].copy()
        engine_samples.append({
            'time_s':float(sim.data.time),
            'clock_increment_error_s':abs(float(sim.data.time)-previous_time-sim.decimation*sim.model.opt.timestep),
            'actuator_force_max_error_nm':float(np.max(np.abs(actual-expected))),
            'actual_actuator_force_sim_order_nm':actual.tolist(),
            'expected_actuator_force_sim_order_nm':expected.tolist(),
        })
    data = {k:np.asarray(v) for k,v in data.items()}
    np.savez_compressed(trace_path, **data, dof_names=np.array(sim._human_joint_model.names))
    metrics = torque_trace_metrics(**data, lower=sim._human_joint_model.lower.numpy(), upper=sim._human_joint_model.upper.numpy())
    return {"condition":"free root, zero gravity, contact disabled, zero PD targets; not walking",
            'initial_wrist_velocity_rad_s':initial_wrist_velocity_rad_s,
            "metrics":metrics, "expected_substeps":steps*sim.decimation,
            "substep_gate":gate(abs(len(data["time"])-steps*sim.decimation),0,evidence_kind="numerical",required_evidence="numerical"),
            "finite_state": bool(np.isfinite(sim.data.qpos).all() and np.isfinite(sim.data.qvel).all()),
            'engine_control_end_samples':engine_samples,
            'actual_actuator_force_gate':gate(max(s['actuator_force_max_error_nm'] for s in engine_samples),2e-4,
                                            evidence_kind='numerical',required_evidence='numerical'),
            'clock_increment_max_error_s':max(s['clock_increment_error_s'] for s in engine_samples),
            'native_warning_counts':[int(w.number) for w in sim.data.warning],
            "trace_sha256":sha256(trace_path)}


def isaaclab_assay(features, trace_path, steps=30):
    """Native GPU physics accounting and partial reset; no walking controller."""
    import os
    os.environ['PROTOMOTIONS_HUMAN_MODEL_FEATURES']=','.join(features)
    from isaaclab.app import AppLauncher
    launcher=AppLauncher(headless=True,device='cuda:0')
    app=launcher.app;sim=None
    try:
        from protomotions.robot_configs.smpl import SmplRobotConfig
        from protomotions.simulator.isaaclab.config import IsaacLabSimulatorConfig
        from protomotions.simulator.isaaclab.simulator import IsaacLabSimulator
        from protomotions.simulator.base_simulator.simulator_state import ResetState, StateConversion
        from protomotions.components.scene_lib import SceneLib, SceneLibConfig
        from protomotions.components.terrains.terrain import Terrain
        from protomotions.components.terrains.config import TerrainConfig
        device=torch.device('cuda:0');robot=SmplRobotConfig()
        cfg=IsaacLabSimulatorConfig(num_envs=2,headless=True,experiment_name='human_model_validation')
        cfg.sim=robot.simulation_params.isaaclab
        terrain=Terrain(TerrainConfig(num_levels=1,num_terrains=1,map_length=4.,map_width=4.,border_size=1.),num_envs=2,device=device)
        sim=IsaacLabSimulator(cfg,robot,terrain,device,app,SceneLib(SceneLibConfig(),num_envs=2))
        sim._initialize_with_markers(None)
        model=sim._human_joint_model
        initial_activation_allocated=(model._activation is not None and tuple(model._activation.shape)==(2,69,2)) if 'activation' in features else None
        root=sim.get_root_state();dof=sim.get_dof_state()
        reset=ResetState(root_pos=root.root_pos.clone(),root_rot=root.root_rot.clone(),
            root_vel=torch.zeros_like(root.root_vel),root_ang_vel=torch.zeros_like(root.root_ang_vel),
            dof_pos=dof.dof_pos.clone(),dof_vel=torch.zeros_like(dof.dof_vel),state_conversion=StateConversion.COMMON)
        reset.root_pos[:,2]=3.
        sim.reset_envs(reset)
        data={key:[] for key in ('time','q','qd','requested','active','elastic','damping','applied','negative_cap','positive_cap')}
        def record(s,state):
            data['time'].append(float(s._sim.current_time))
            for key,value in (('q',state.dof_pos),('qd',state.dof_vel),
                ('requested',s.human_requested_torques),('active',s.human_active_torques),
                ('elastic',s.human_elastic_torques),('damping',s.human_damping_torques),
                ('applied',s.human_applied_torques),('negative_cap',s.human_negative_caps),('positive_cap',s.human_positive_caps)):
                data[key].append(value[0].detach().cpu().numpy().copy())
        sim.human_model_trace_callback=record
        start_index=sim._sim.current_time_step_index
        for _ in range(steps):
            sim.step(torch.zeros((2,robot.number_of_actions),device=device))
        physical_steps=sim._sim.current_time_step_index-start_index
        state=sim.get_dof_state()
        finite=bool(torch.isfinite(state.dof_pos).all() and torch.isfinite(state.dof_vel).all())
        partial_reset=None
        if 'activation' in features:
            model._activation.fill_(.3)
            root=sim.get_root_state();dof=sim.get_dof_state()
            one=ResetState(root_pos=root.root_pos[1:].clone(),root_rot=root.root_rot[1:].clone(),
                root_vel=root.root_vel[1:].clone(),root_ang_vel=root.root_ang_vel[1:].clone(),
                dof_pos=dof.dof_pos[1:].clone(),dof_vel=dof.dof_vel[1:].clone(),state_conversion=StateConversion.COMMON)
            sim.reset_envs(one,env_ids=torch.tensor([1],device=device))
            partial_reset=bool((model._activation[1]==0).all() and (model._activation[0]==.3).all())
        data={k:np.asarray(v) for k,v in data.items()}
        np.savez_compressed(trace_path,**data,dof_names=np.array(model.names))
        result={'condition':'2 free-root environments, root z=3m, gravity and terrain enabled, zero actions; env0 trace; no walking controller',
                'metrics':torque_trace_metrics(**data,lower=model.lower.cpu().numpy(),upper=model.upper.cpu().numpy()),
                'physical_steps':physical_steps,'expected_substeps':steps*sim.decimation,
                'substep_gate':gate(max(abs(len(data['time'])-physical_steps),abs(physical_steps-steps*sim.decimation)),0,
                    evidence_kind='numerical',required_evidence='numerical'),
                'finite_state':finite,'initial_activation_allocated':initial_activation_allocated,
                'partial_activation_reset':partial_reset,'trace_sha256':sha256(trace_path),
                'physics_dt_s':sim._sim.get_physics_dt(),'gpu':torch.cuda.get_device_name(0)}
        # Preserve completed measurements before Kit shutdown, which can stall.
        # The enclosing run remains incomplete until close returns successfully.
        checkpoint=Path(trace_path).with_name(Path(trace_path).stem+'_backend_checkpoint.json')
        checkpoint.write_text(json.dumps({'implementation_id':model.profile_id,'status':'candidate',
            'scope':'Completed backend measurements; shutdown pending','backend':result},indent=2,allow_nan=False)+'\n')
        return result
    finally:
        if sim is not None:
            from protomotions.simulator.base_simulator.simulator import Simulator
            Simulator.close(sim)
        import faulthandler
        faulthandler.dump_traceback_later(30,repeat=False)
        try:
            app.close(wait_for_replicator=False)
        finally:
            faulthandler.cancel_dump_traceback_later()


def video_engine_checks(data, metrics, reset_checks, fixed_base, physics_fps=480):
    """Engineering gates for this complete fixed-base assay, not biological fit."""
    arrays={key:np.asarray(value) for key,value in data.items()}
    phase=arrays['phase']
    hold=np.isin(phase,['hold','reset and hold'])
    sweep=phase=='joint sweep'
    saturation=phase=='torque saturation'
    def check(value, threshold):
        return {'value':float(value) if np.isfinite(value) else None,'threshold_max':threshold,'passed':bool(np.isfinite(value) and value<=threshold)}
    checks={
        'rom_excess_rad':check(np.deg2rad(max(*metrics['rom_excess_max_deg'],metrics.get('terminal_rom_excess_max_deg',0))),1e-4),
        'active_cap_excess_nm':check(max(metrics['cap_violation_max_nm']),2e-4),
        'torque_sum_error_nm':check(max(metrics['torque_accounting_max_nm']),2e-4),
        'actuator_buffer_error_nm':check(np.abs(arrays['engine']-arrays['applied']).max(),2e-4),
        'damping_energy_injection_j':check(max(metrics['damping_energy_injection_j']),1e-6),
        'hold_position_rad':check(np.abs(arrays['q'][hold]).max() if hold.any() else np.inf,math.radians(.5)),
        'hold_velocity_rad_s':check(np.abs(arrays['qd'][hold]).max() if hold.any() else np.inf,.02),
        'finite_state':{'passed':bool(np.isfinite(arrays['q']).all() and np.isfinite(arrays['qd']).all())},
        'fixed_base':{'passed':bool(fixed_base)},
        'complete_phases':{'passed':bool(np.array_equal(phase,np.repeat(
            ['hold','joint sweep','passive release','torque saturation','reset and hold'],[60,180,120,60,60])))},
        'physics_sampling':{'passed':metrics['n_substeps']==len(phase)*(physics_fps//30) and bool(metrics.get('sampling_grid',{}).get('passed',False))},
        'frame_timing':check(np.max(np.abs(np.diff(arrays['time'])-1/30)),1e-5),
        'reset':{'passed':len(reset_checks)==1 and all(row['q_error_rad']<=1e-6 and row['qd_error_rad_s']<=1e-6 for row in reset_checks)},
        'motion_exercised':{'passed':bool(sweep.any() and np.ptp(arrays['q'][sweep],axis=0).max()>math.radians(20))},
        'saturation_exercised':{'passed':bool(saturation.any() and
            (arrays['requested'][saturation]>200).any() and (arrays['requested'][saturation]<-200).any() and
            (arrays['active'][saturation]>0).any() and (arrays['active'][saturation]<0).any() and max(metrics['saturation_fraction'])>0)},
    }
    return {'scope':'fixed-base L0/L1 engineering assay; biological validation separate',
            'status':'passed' if all(row['passed'] for row in checks.values()) else 'candidate_failed_gate','checks':checks}


def fixed_base_inertia_audit(matrix, joint_names, joint_positions, source):
    """Compare a native joint matrix with MJCF at an explicitly supplied pose.

    This diagnostic reports both authored and zero-armature references. It does
    not infer the simulator's armature or grant a dynamics equivalence gate.
    """
    import mujoco
    matrix = np.asarray(matrix, dtype=float)
    positions = np.asarray(joint_positions, dtype=float)
    names = list(joint_names)
    if (not names or len(set(names)) != len(names)
            or matrix.shape != (len(names), len(names))
            or positions.shape != (len(names),)
            or not np.isfinite(matrix).all() or not np.isfinite(positions).all()):
        raise ValueError('Finite square matrix, unique joint names and matching pose required')
    reference = mujoco.MjModel.from_xml_path(str(source))
    state = mujoco.MjData(reference)
    joint_ids = [mujoco.mj_name2id(reference, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
    if any(i < 0 or reference.jnt_type[i] != mujoco.mjtJoint.mjJNT_HINGE for i in joint_ids):
        raise ValueError('Every supplied joint must name an MJCF hinge')
    addresses = reference.jnt_dofadr[joint_ids]
    state.qpos[reference.jnt_qposadr[joint_ids]] = positions
    authored_armature = reference.dof_armature[addresses].copy()
    comparisons = {}
    for label in ('authored_armature', 'zero_armature'):
        if label == 'zero_armature':
            reference.dof_armature[addresses] = 0
        mujoco.mj_forward(reference, state)
        full = np.empty((reference.nv, reference.nv))
        mujoco.mj_fullM(reference, state, full)
        expected = full[np.ix_(addresses, addresses)]
        error = matrix - expected
        comparisons[label] = dict(
            max_abs_kg_m2=float(np.abs(error).max()),
            max_off_diagonal_abs_kg_m2=float(np.abs(error-np.diag(np.diag(error))).max()),
            relative_frobenius=float(np.linalg.norm(error)/np.linalg.norm(expected)),
            diagonal_error_range_kg_m2=[float(np.diag(error).min()), float(np.diag(error).max())])
    return dict(status='diagnostic', scope='fixed base, supplied pose only; no dynamics or biological gate',
                source=str(source), source_sha256=sha256(source), joint_names=names,
                joint_positions_rad=positions.tolist(), authored_armature_kg_m2=authored_armature.tolist(),
                symmetry_error_kg_m2=float(np.abs(matrix-matrix.T).max()),
                minimum_symmetric_eigenvalue_kg_m2=float(np.linalg.eigvalsh((matrix+matrix.T)/2).min()),
                comparisons=comparisons)


def isaaclab_kinematic_audit(sim, initial, model):
    """Independent MuJoCo FK/mass comparison of the actual initialized asset."""
    import mujoco
    from scipy.spatial.transform import Rotation
    from protomotions.simulator.base_simulator.simulator_state import ResetState, StateConversion
    source=(PACKAGE_ROOT/'human_model_v2/assets/human_model_v2.xml') if model.profile_id=='human_model_v2' else ASSET_ROOT/'mjcf/smpl_humanoid.xml'
    reference=mujoco.MjModel.from_xml_path(str(source));state=mujoco.MjData(reference)
    names=sim.robot_config.kinematic_info.body_names
    body_ids=[mujoco.mj_name2id(reference,mujoco.mjtObj.mjOBJ_BODY,n) for n in names]
    addresses=[reference.jnt_qposadr[mujoco.mj_name2id(reference,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in model.names]
    fractions=np.random.default_rng(481).uniform(.2,.8,(3,len(model.names)))
    poses=np.concatenate([np.zeros((1,len(model.names))),model.lower.cpu().numpy()+fractions*(model.upper-model.lower).cpu().numpy()])
    errors=[]
    for pose in poses:
        reset=ResetState(root_pos=initial.root_pos,root_rot=initial.root_rot,root_vel=initial.root_vel,root_ang_vel=initial.root_ang_vel,dof_pos=torch.tensor(pose[None],device=sim.device,dtype=model.lower.dtype),dof_vel=initial.dof_vel,state_conversion=StateConversion.COMMON)
        sim.reset_envs(reset)
        actual=sim.get_bodies_state()
        state.qpos[:3]=initial.root_pos[0].cpu().numpy()
        state.qpos[3:7]=initial.root_rot[0].cpu().numpy()[[3,0,1,2]]
        state.qpos[addresses]=pose;mujoco.mj_forward(reference,state)
        positions=actual.rigid_body_pos[0].cpu().numpy()
        rotations=Rotation.from_quat(actual.rigid_body_rot[0].cpu().numpy()).as_matrix()
        errors.append(dict(position_max_m=float(np.abs(positions-state.xpos[body_ids]).max()),rotation_matrix_max=float(np.abs(rotations-state.xmat[body_ids].reshape(-1,3,3)).max())))
    native_ids=[sim._robot.body_names.index(n) for n in names]
    view=sim._robot.root_physx_view
    masses=view.get_masses()[0,native_ids].cpu().numpy()
    com=view.get_coms()[0,native_ids,:3].cpu().numpy()
    inertia=view.get_inertias()[0,native_ids].cpu().numpy().reshape(-1,3,3)
    axes=Rotation.from_quat(reference.body_iquat[body_ids][:,[1,2,3,0]]).as_matrix()
    expected=(axes*reference.body_inertia[body_ids,None,:])@axes.transpose(0,2,1)
    mass_errors=dict(mass_max_kg=float(abs(masses-reference.body_mass[body_ids]).max()),com_max_m=float(abs(com-reference.body_ipos[body_ids]).max()),inertia_max_kg_m2=float(abs(inertia-expected).max()))
    sim.reset_envs(initial)
    # Diagnostic only: distinguish the matrix API from the inertia used by
    # actual stepping. No acceptance threshold is inferred from these samples.
    order=sim.data_conversion.dof_convert_to_common
    armature=view.get_dof_armatures()[0].cpu().numpy()
    response=[]
    for torque in (-.01,.01):
        sim.reset_envs(initial)
        matrix=view.get_generalized_mass_matrices()[0].cpu().numpy().astype(float)
        requested=torch.zeros_like(initial.dof_pos)
        requested[0,model.names.index('R_Wrist_x')]=torque
        before=sim.get_dof_state().dof_vel.clone()
        sim._common_actions=requested
        sim._apply_control();sim._scene.write_data_to_sim()
        applied=sim._robot.data.applied_torque[0].cpu().numpy().astype(float).copy()
        sim._sim.step(render=False);sim._scene.update(dt=sim._sim.get_physics_dt())
        acceleration=((sim.get_dof_state().dof_vel-before)/sim._sim.get_physics_dt())[0]
        # Convert measured common-order acceleration back to native order.
        native_acceleration=torch.empty_like(acceleration);native_acceleration[order]=acceleration
        actual=native_acceleration.cpu().numpy().astype(float)
        plain=np.linalg.solve(matrix,applied)
        augmented=np.linalg.solve(matrix+np.diag(armature),applied)
        response.append(dict(request_nm=torque,applied_native_nm=applied.tolist(),
            measured_acceleration_native_rad_s2=actual.tolist(),
            matrix_only_prediction_rad_s2=plain.tolist(),
            matrix_plus_armature_prediction_rad_s2=augmented.tolist(),
            matrix_only_max_error_rad_s2=float(abs(actual-plain).max()),
            matrix_plus_armature_max_error_rad_s2=float(abs(actual-augmented).max())))
    inertia_response=dict(status='diagnostic',condition='neutral pose, zero velocity, +/-0.01Nm right wrist x, one physics step; constraints retained',
        native_joint_names=list(sim._robot.joint_names),native_armature_kg_m2=armature.tolist(),
        dt_s=sim._sim.get_physics_dt(),trials=response)
    sim.reset_envs(initial)
    passed=all(e['position_max_m']<=1e-5 and e['rotation_matrix_max']<=1e-5 for e in errors) and mass_errors['mass_max_kg']<=1e-5 and mass_errors['com_max_m']<=1e-6 and mass_errors['inertia_max_kg_m2']<=1e-6
    return dict(status='passed' if passed else 'candidate_failed_gate',source=str(source),source_sha256=sha256(source),body_count=len(names),poses_rad=poses.tolist(),pose_errors=errors,mass_errors=mass_errors,inertia_response=inertia_response,scope='24 anatomical links; numerical frame masses accounted separately; inertia response diagnostic excluded from gate')


def joint_demo_plan(scenario):
    """Gentle independent-axis chapters; references are bounds, not trajectories."""
    plans = {
        'lower_body': [('Hip', 'R_Hip_y', -40, 8), ('Knee', 'R_Knee_y', 50, 0),
                       ('Ankle', 'R_Ankle_y', -8, 20), ('Toe', 'R_Toe_y', -20, 15)],
        'upper_body': [('Shoulder', 'R_Shoulder_x', 40, -20),
                       ('Elbow', 'R_Elbow_z', 50, 0), ('Wrist', 'R_Wrist_x', 20, -20)],
    }
    if scenario not in plans:
        raise ValueError('Unknown joint demonstration scenario')
    return [dict(label=a, joint=b, positive_waypoint_deg=c, negative_waypoint_deg=d,
                 start_s=i*8, duration_s=8) for i,(a,b,c,d) in enumerate(plans[scenario])]


def joint_demo_target(chapter, local_time):
    """C2 quintic interpolation, zero velocity/acceleration at every waypoint."""
    knots=[0., 2.5, 5.5, 8.]
    values=[0.,chapter['positive_waypoint_deg'],chapter['negative_waypoint_deg'],0.]
    t=float(np.clip(local_time,0,8))
    i=min(int(np.searchsorted(knots,t,side='right')-1),2)
    duration=knots[i+1]-knots[i];u=(t-knots[i])/duration
    delta=math.radians(values[i+1]-values[i]);start=math.radians(values[i])
    return (start+delta*(10*u**3-15*u**4+6*u**5),
            delta*(30*u**2-60*u**3+30*u**4)/duration,
            delta*(60*u-180*u**2+120*u**3)/duration**2)


def population_fixture_target(chapter, local_time):
    """Smooth external measurement rig trajectory, degrees in config, SI out."""
    if chapter.get('rom_suite'):
        knots=chapter['trajectory_knots']
        for (ta,qa),(tb,qb) in zip(knots,knots[1:]):
            if local_time<=tb:
                u=float(np.clip((local_time-ta)/(tb-ta),0,1));duration=tb-ta
                return tuple(map(math.radians,(qa+(qb-qa)*(10*u**3-15*u**4+6*u**5),
                    (qb-qa)*(30*u*u-60*u**3+30*u**4)/duration,
                    (qb-qa)*(60*u-180*u*u+120*u**3)/duration**2)))
        return math.radians(knots[-1][1]),0.,0.
    start=chapter['start_deg'];delta=chapter['end_deg']-start
    elapsed=local_time-2
    if chapter.get('kind')=='harmonic':
        duration=chapter['duration_s']-4
        if elapsed<=0 or elapsed>=duration:return math.radians(start),0.,0.
        envelope=1.;first=second=0.
        ramp=1.
        if elapsed<ramp or elapsed>duration-ramp:
            falling=elapsed>duration-ramp
            u=(duration-elapsed if falling else elapsed)/ramp
            envelope=10*u**3-15*u**4+6*u**5
            first=(30*u**2-60*u**3+30*u**4)/ramp*(-1 if falling else 1)
            second=(60*u-180*u**2+120*u**3)/ramp**2
        omega=2*math.pi*chapter['frequency_hz'];phase=omega*elapsed
        amplitude=math.radians(chapter['amplitude_deg']);s=math.sin(phase);c=math.cos(phase)
        return (math.radians(start)+amplitude*envelope*s,
            amplitude*(first*s+envelope*omega*c),
            amplitude*(second*s+2*first*omega*c-envelope*omega*omega*s))
    if 'speed_deg_s' in chapter and delta:
        speed=chapter['speed_deg_s'];ramp=chapter.get('speed_ramp_s',1.)
        if speed<=0 or ramp<=0 or abs(delta)<speed*ramp:raise ValueError('Invalid constant-speed fixture ramp')
        duration=abs(delta)/speed+ramp;sign=1 if delta>0 else -1
        if elapsed<=0:return math.radians(start),0.,0.
        if elapsed>=duration:return math.radians(start+delta),0.,0.
        if elapsed<ramp:
            u=elapsed/ramp;distance=speed*ramp*(u**3-.5*u**4)
            velocity=speed*(3*u*u-2*u**3);acceleration=speed/ramp*(6*u-6*u*u)
        elif elapsed>duration-ramp:
            u=(duration-elapsed)/ramp;distance=abs(delta)-speed*ramp*(u**3-.5*u**4)
            velocity=speed*(3*u*u-2*u**3);acceleration=-speed/ramp*(6*u-6*u*u)
        else:distance=speed*(elapsed-ramp/2);velocity=speed;acceleration=0.
        return tuple(map(math.radians,(start+sign*distance,sign*velocity,sign*acceleration)))
    duration=chapter.get('ramp_s',chapter['duration_s']-4)
    u=np.clip(elapsed/duration,0.,1.)
    return (math.radians(start+delta*(10*u**3-15*u**4+6*u**5)),
            math.radians(delta*(30*u**2-60*u**3+30*u**4)/duration) if 0<u<1 else 0.,
            math.radians(delta*(60*u-180*u**2+120*u**3)/duration**2) if 0<u<1 else 0.)


def active_rom_drive(chapter, focus, q, qd, desired, velocity, feedforward, mass, gravity, fixture):
    """Diagnostic torque PD; all assistance to the tested axis uses human caps.

    The native fixed-base rig holds other coordinates. This is not a trained
    policy or an active-ROM population reference. Never leave fixture gravity
    compensation on the test axis.
    """
    kp = float(chapter['active_kp_nm_rad'])
    if not math.isfinite(kp) or kp <= 0:
        raise ValueError('Active ROM requires finite positive torque PD stiffness')
    inertia = mass[:, focus, focus]
    kd = 2 * torch.sqrt(kp * inertia)
    command = torch.zeros_like(q)
    command[:, focus] = (kp * (desired-q[:, focus]) + kd * (velocity-qd[:, focus])
                         + inertia * feedforward + gravity[:, focus])
    fixture = fixture.clone()
    fixture[:, focus] = 0
    return command, fixture


def all_direction_rom_protocols(profile):
    """Existing 69 coordinates, both directions; fixed parameters, no plant tuning.

    Locked axes receive a separately labelled external 0.5 Nm constraint probe.
    Source ROM is a model bound, not an independent active-ROM population band.
    The fixture/gain recipe was confirmed with the population candidate. Callers
    must attach its profile_path when using a non-packaged profile; an arbitrary
    new profile still requires its own native confirmation.
    """
    groups={key:[] for key in ('lower','feet','upper','axial')}
    for joint,row in profile['joints'].items():
        body=joint.rsplit('_',1)[0];region=body.removeprefix('L_').removeprefix('R_')
        axis=joint[-1]
        families={'Hip':{'x':'adduction / abduction','y':'flexion / extension','z':'internal / external rotation'},
            'Knee':{'y':'extension / flexion'},'Ankle':{'x':'inversion / eversion','y':'dorsiflexion / plantarflexion'},
            'Toe':{'y':'MTP rotation proxy'},'Shoulder':{'x':'arm lowering / elevation','y':'humeral rotation','z':'flexion / extension'},
            'Elbow':{'z':'extension / flexion'},'Wrist':{'x':'flexion / extension','y':'pronation / supination proxy','z':'radial / ulnar deviation'}}
        family=families.get(region,{}).get(axis,'near-lock axis' if max(row['active_nm'])==0 else {'x':'lateral bend / girdle proxy','y':'flexion / extension proxy','z':'axial rotation / girdle proxy'}[axis])
        group='lower' if region in ('Hip','Knee') else 'feet' if region in ('Ankle','Toe') else 'upper' if region in ('Shoulder','Elbow','Wrist','Hand','Thorax') else 'axial'
        lo,hi=row['rom_deg'];locked=max(row['active_nm'])==0
        neutral=float(np.clip(0,lo,hi))
        for direction,bound in ((-1,lo),(1,hi)):
            end=bound-direction*min(.1,(hi-lo)/10)
            ramp=round(max(1.5,1.875*abs(end-neutral)/25)*30)/30
            hold_start=1+ramp;probe_start=hold_start+1.5
            challenge=bound+direction*(.5 if locked else 3)
            return_start=probe_start+1.5
            duration=return_start+ramp+1.5
            pose={}
            if joint.endswith('Hip_y') and direction<0:pose[joint.replace('Hip_y','Knee_y')]=90
            if joint.endswith('Knee_y'):pose[joint.replace('Knee_y','Hip_y')]=-90
            if joint=='L_Hip_x' and direction<0:pose['R_Hip_x']=-30
            if joint=='R_Hip_x' and direction>0:pose['L_Hip_x']=30
            kp=500. if joint in ('Torso_z','Spine_z','Chest_z') else 2000.
            if joint.endswith('Hip_y') and direction<0:kp=2500.
            groups[group].append(dict(label=f'{joint} {"negative" if direction<0 else "positive"} / {"LOCK probe" if locked else "active ROM"}',
                joint=joint,kind='active_rom',rom_suite=True,locked_axis=locked,direction=direction,
                pose_deg=pose,motion_family=family,
                start_deg=neutral,end_deg=end,duration_s=duration,active_kp_nm_rad=kp,
                endpoint_window_s=[hold_start+.5,probe_start],probe_window_s=[probe_start+.5,return_start],
                recovery_window_s=[duration-1,duration],rom_deg=[lo,hi],
                trajectory_knots=[[0,neutral],[1,neutral],[hold_start,end],[probe_start,end],
                    [probe_start+.5,challenge],[return_start,challenge],[return_start+ramp,neutral],[duration,neutral]],
                external_probe_nm=.5 if locked else 0.,
                reference_note='Configured ROM only; no matched independent active-ROM cohort',
                source_metadata={k:v for k,v in row.items() if k not in ('stiffness_nm_per_rad','damping_nms_per_rad')}))
    return {key:dict(assay='all_direction_active_rom',video_number=key.upper(),root_pos=[3,3,1.6],
        pose_deg={'L_Shoulder_x':-90,'R_Shoulder_x':90},chapters=chapters,fixture_acceleration_gain=64000.,
        support_mode='joint_clamp',physics_device='cpu',preallocated_trace=True,
        scope='Fixed pelvis and other axes; gravity and terrain ON; limbs elevated; self-collision OFF; both directions including zero-strength axes',
        engineering_thresholds={'endpoint_error_deg':2.,'rom_excess_deg':1.,'hold_peak_to_peak_deg':1.,
            'recovery_error_deg':2.,'non_test_error_deg':2.,'cap_error_nm':2e-4,'torque_sum_error_nm':2e-4},
        selection_status='candidate') for key,chapters in groups.items()}


def suspended_gait_target(names, t, *, duration=12., ramp=2., frequency=.4):
    """C2-envelope synthetic limb targets (q, qdot, qddot); not human data."""
    if duration <= 2*ramp or ramp <= 0 or frequency <= 0:
        raise ValueError('Invalid suspended gait timing')
    def smooth(x):
        if x <= 0:return 0.,0.,0.
        if x >= 1:return 1.,0.,0.
        return 10*x**3-15*x**4+6*x**5,30*x*x*(1-x)**2,60*x-180*x*x+120*x**3
    a,ad,add=smooth(t/ramp);b,bd,bdd=smooth((duration-t)/ramp)
    ad/=ramp;add/=ramp*ramp;bd/=-ramp;bdd/=ramp*ramp
    e,ed,edd=a*b,ad*b+a*bd,add*b+2*ad*bd+a*bdd
    q=np.zeros(len(names));v=q.copy();acc=q.copy();w=2*math.pi*frequency
    def coord(name,base,offset,amplitude,phase):
        i=names.index(name);phi=w*t+phase
        f=offset+amplitude*math.sin(phi);fd=amplitude*w*math.cos(phi);fdd=-amplitude*w*w*math.sin(phi)
        q[i]=base+e*f;v[i]=ed*f+e*fd;acc[i]=edd*f+2*ed*fd+e*fdd
    for side,phase,sign in [('L',0.,-1.),('R',math.pi,1.)]:
        coord(side+'_Hip_y',-5,0,-18,phase)
        coord(side+'_Knee_y',8,16,16,phase-.55)
        coord(side+'_Ankle_y',2,0,7,phase+.4)
        coord(side+'_Toe_y',0,0,10,phase+1.)
        coord(side+'_Shoulder_x',sign*80,0,0,0)
        coord(side+'_Shoulder_z',0,0,-sign*15,phase)
        coord(side+'_Elbow_z',sign*18,0,sign*5,phase+.5)
    for side,phase in [('L',0.),('R',math.pi)]:
        if side+'_Subtalar_x' in names:coord(side+'_Subtalar_x',0,0,4,phase+.6)
    return tuple(np.deg2rad(x) for x in (q,v,acc))


def isaaclab_validation_video(output, seed, *, render=True, frames=480, physics_fps=480, start_frame=0, position_iterations=32, velocity_iterations=1, external_forces_every_iteration=False, solver_type=1, joint_mode='serial', frame_mass=1e-6, audit_kinematics=False, scenario="stress", population_config=None):
    """L0/L1 fixed-base diagnostic: native rendered frames and matched measurements."""
    suspended_gait=scenario=='suspended_gait'
    v2_suite=scenario=='v2_suite'
    population_load=scenario=='population_load'
    population_joint=scenario=='population_joint'
    protocol=json.loads(Path(population_config).read_text()) if population_config else None
    model_tag=protocol.get('model_id','healthy_adult_v1') if protocol else 'healthy_adult_v1'
    if protocol and protocol.get('cpu_force_evaluator',False):
        raise ValueError('Archived mixed CPU-force/GPU-physics assay: reproduce using its source bundle')
    chapters=joint_demo_plan(scenario) if scenario not in ('stress','population_load','population_joint','suspended_gait','v2_suite') else None
    if suspended_gait:
        frames=round((protocol['duration_s']+protocol.get('settle_s',0))*30)
    if population_joint:
        chapters=protocol['chapters']
        for i,c in enumerate(chapters):
            c['start_s']=sum(x['duration_s'] for x in chapters[:i])
        frames=round(sum(c['duration_s'] for c in chapters)*30)
    if chapters is not None and not population_joint:
        frames=240*len(chapters)
        if start_frame:raise ValueError("Chapter demos start at frame zero")
    import os
    import imageio.v2 as imageio
    os.environ['PROTOMOTIONS_HUMAN_MODEL_FEATURES']=''
    from isaaclab.app import AppLauncher
    physics_device=protocol.get('physics_device','cuda:0') if protocol else 'cuda:0'
    launcher=AppLauncher(headless=True,enable_cameras=render,device=physics_device)
    app=launcher.app
    from protomotions.robot_configs.smpl import SmplRobotConfig
    from protomotions.robot_configs.base import ControlType
    from protomotions.simulator.isaaclab.config import IsaacLabSimulatorConfig
    from protomotions.simulator.isaaclab.simulator import IsaacLabSimulator
    from protomotions.components.scene_lib import SceneLib, SceneLibConfig
    from protomotions.components.terrains.terrain import Terrain
    from protomotions.components.terrains.config import TerrainConfig
    from protomotions.simulator.base_simulator.simulator_state import ResetState, StateConversion
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    assay_name='v2_suite' if v2_suite else 'suspended_gait' if suspended_gait else 'population_joint_gravity_contact' if population_joint else ('population_load_gravity_contact' if population_load else 'l0_l1_fixedbase')
    raw=output/f'{model_tag}_{assay_name}_seed{seed}_raw.mp4'
    trace=raw.with_suffix('.npz')
    if raw.exists() or trace.exists():raise FileExistsError(raw)
    if protocol and protocol.get("model_id")=="human_model_v2":
        from ..human_model_v2.model_config import robot_config
        robot=robot_config();joint_mode="anatomical"
    else:robot=SmplRobotConfig()
    if protocol and 'human_model_parameters' in protocol:
        robot.human_model_parameters=protocol['human_model_parameters']
    robot.asset.fix_base_link=True;robot.asset.disable_gravity=True
    if population_load:
        robot.asset.fix_base_link=False;robot.asset.disable_gravity=False
        robot.contact_bodies=list(robot.kinematic_info.body_names)
    if population_joint or suspended_gait or v2_suite:
        robot.asset.disable_gravity=False
        robot.contact_bodies=list(robot.kinematic_info.body_names)
    if population_load or population_joint or suspended_gait:
        robot.validation_contact_capacity=128
    robot.human_model_usd_joint_mode=joint_mode;robot.human_model_joint_frame_mass=frame_mass
    if protocol and protocol.get("assay") in ("supported_feet", "smpl2hm_tracking"):robot.asset.fix_base_link=False
    robot.asset.self_collisions=False;robot.default_root_height=1.5
    robot.control.control_type=ControlType.TORQUE
    cfg=IsaacLabSimulatorConfig(num_envs=protocol.get('num_envs',1) if protocol else 1,headless=True,experiment_name='human_model_l0_l1_video')
    if physics_fps<=0 or physics_fps%30:raise ValueError('physics_fps must be a positive multiple of 30')
    cfg.sim=robot.simulation_params.isaaclab;cfg.sim.fps=physics_fps;cfg.sim.decimation=physics_fps//30;cfg.record_viewer=render
    cfg.sim.physx.num_position_iterations=position_iterations;cfg.sim.physx.num_velocity_iterations=velocity_iterations
    cfg.sim.physx.solver_type=solver_type
    device=torch.device(physics_device)
    terrain_size=max(4.,4.*math.ceil(math.sqrt(cfg.num_envs)),float(protocol.get("terrain_size",0)) if protocol else 0.)
    terrain=Terrain(TerrainConfig(num_levels=1,num_terrains=1,map_length=terrain_size,map_width=terrain_size,border_size=1.),num_envs=cfg.num_envs,device=device)
    sim=IsaacLabSimulator(cfg,robot,terrain,device,app,SceneLib(SceneLibConfig()))
    if population_joint:
        # Visualize the already-declared fixed-pelvis constraint. These blue
        # shapes are the rig drawing, not additional hidden contact supports.
        from pxr import UsdGeom,Gf
        x,y,z=protocol['root_pos'];seat_top=z-.0843
        for label,position,size in [
            ('seat',[x-.06,y,seat_top-.025],[.34,.38,.05]),
            ('post',[x-.06,y,(seat_top-.05)/2],[.08,.08,seat_top-.05]),
            ('base',[x-.06,y,.015],[.55,.55,.03])]:
            cube=UsdGeom.Cube.Define(sim._sim.stage,'/World/dynamometer_'+label)
            cube.CreateSizeAttr(1.)
            UsdGeom.XformCommonAPI(cube).SetTranslate(Gf.Vec3d(*position))
            UsdGeom.XformCommonAPI(cube).SetScale(Gf.Vec3f(*size))
            cube.CreateDisplayColorAttr([Gf.Vec3f(.12,.35,.6)])
    if external_forces_every_iteration:
        from pxr import PhysxSchema, UsdPhysics
        for prim in sim._sim.stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                PhysxSchema.PhysxSceneAPI.Apply(prim).CreateEnableExternalForcesEveryIterationAttr().Set(True)
    sim._initialize_with_markers(None)
    model=sim._human_joint_model;k=model.names.index('R_Knee_y');w=model.names.index('R_Wrist_x')
    if population_joint and protocol.get('profile_path'):
        from .profile import validate_profile
        candidate_profile=validate_profile(json.loads(Path(protocol['profile_path']).read_text()))
        raw.with_name(raw.stem+'_profile.json').write_text(json.dumps(candidate_profile,indent=2))
        # Local experiment candidate only; packaged/checkpoint profile unchanged.
        model=HumanJointModel(candidate_profile,model.names,device=device)
        sim._human_joint_model=model
    root=sim.get_root_state();dof=sim.get_dof_state()
    initial=ResetState(root_pos=root.root_pos.clone(),root_rot=root.root_rot.clone(),root_vel=root.root_vel*0,root_ang_vel=root.root_ang_vel*0,dof_pos=dof.dof_pos*0,dof_vel=dof.dof_vel*0,state_conversion=StateConversion.COMMON)
    if population_load:
        # Terrain mesh occupies positive XY; origin lies on its corner.
        initial.root_pos[:]=torch.tensor([3.,3.,.24],device=device)
        # COMMON quaternion uses xyzw: turn upright SMPL into supine posture.
        initial.root_rot[:]=torch.tensor([0.,-math.sqrt(.5),0.,math.sqrt(.5)],device=device)
    if population_joint:
        initial.root_pos[:]=torch.tensor(protocol['root_pos'],device=device)
        initial.root_rot[:]=torch.tensor(protocol.get('root_rot_xyzw',[0,0,0,1]),device=device)
        for name,angle in protocol['pose_deg'].items():initial.dof_pos[:,model.names.index(name)]=math.radians(angle)
    if suspended_gait:
        initial.root_pos[:]=torch.tensor([3.,3.,1.5],device=device)
        initial.root_rot[:]=torch.tensor([0.,0.,0.,1.],device=device)
        initial.dof_pos[:]=torch.as_tensor(suspended_gait_target(model.names,0.,duration=protocol['duration_s'])[0],device=device,dtype=initial.dof_pos.dtype)
    fixture_pose_common=initial.dof_pos.clone()
    population_base_pose_common=fixture_pose_common.clone()
    sim.reset_envs(initial)
    if v2_suite:
        from ..human_model_v2.validation import native_suite
        return native_suite(sim,app,output,seed,protocol,initial)
    reset_checks=[]
    kinematic_audit=isaaclab_kinematic_audit(sim,initial,model) if audit_kinematics else None
    if kinematic_audit is not None:
        raw.with_name(raw.stem+'_kinematic_audit.json').write_text(json.dumps(kinematic_audit,indent=2))
        print('KINEMATIC_AUDIT',kinematic_audit['status'],kinematic_audit['pose_errors'],kinematic_audit['mass_errors'],flush=True)
    pv=None
    if render:
        from protomotions.simulator.isaaclab.utils.perspective_viewer import PerspectiveViewer
        pv=PerspectiveViewer(resolution=(720,720));pv.set_camera_view([2.4,-2.4,2.0],[0,0,1.25])
        if suspended_gait:pv.set_camera_view([5.8,.1,2.0],[3.,3.,1.35])
        if population_load:pv.set_camera_view([4.6,.7,1.7],[3,3,.2])
        for _ in range(12):sim._sim.render()
    data={key:[] for key in ('time','q','qd','requested','active','elastic','damping','applied','engine','phase','target')}
    if suspended_gait:
        data.update({key:[] for key in ('root_pos','native_effort','gravity','coriolis')})
    if population_load:
        data.update({key:[] for key in ('contact_force_w','com_w','com_velocity_w','root_pos','incoming_joint_wrench')})
    if population_joint:
        data.update({key:[] for key in ('fixture','engine_total','gravity','coriolis','inertia_accel','projected_joint_force','contact_force_w','chapter_index')})
    if population_load or population_joint or suspended_gait:
        data.update({key:[] for key in ('contact_body_forces_w','contact_min_separation_m','contact_points_count')})
    start=sim._sim.current_time
    mass=sim._robot.root_physx_view.get_generalized_mass_matrices()
    view=sim._robot.root_physx_view
    np.savez_compressed(raw.with_name(raw.stem+'_native_initial.npz'),mass_matrix=mass.cpu().numpy(),armature=view.get_dof_armatures().cpu().numpy(),coms=view.get_coms().cpu().numpy(),masses=view.get_masses().cpu().numpy(),inertias=view.get_inertias().cpu().numpy(),joint_limits=sim._robot.data.joint_pos_limits.cpu().numpy(),joint_names=np.asarray(sim._robot.joint_names),body_names=np.asarray(sim._robot.body_names))
    base_offset=0 if sim._robot.is_fixed_base else 6
    inertia=mass[:,base_offset:,base_offset:].diagonal(dim1=-2,dim2=-1)[:,sim.data_conversion.dof_convert_to_common].clamp_min(1e-6)
    print('FIXED_BASE',sim._robot.is_fixed_base,'MASS_RANGE',float(inertia.min()),float(inertia.max()),flush=True)
    order=sim.data_conversion.dof_convert_to_common
    fixture_armature=sim._robot.root_physx_view.get_dof_armatures().to(device=sim.device)[:,order].clone()
    substeps={key:[] for key in ('time','q','qd','requested','active','elastic','damping','applied','negative_cap','positive_cap')}
    preallocated_trace=bool(population_joint and protocol.get('preallocated_trace',False))
    trace_cursor=0;trace_capacity=frames*sim.decimation
    if preallocated_trace:
        substeps={key:np.empty(trace_capacity if key=='time' else (trace_capacity,len(model.names)),
            dtype=np.float64 if key=='time' else np.float32) for key in substeps}
    active_audit={key:[] for key in ('time','chapter_index','focus','target_rad','fixture_focus_nm')}
    clamp_limit_audit=[]
    def record_substep(s, state):
        nonlocal trace_cursor
        if preallocated_trace:substeps['time'][trace_cursor]=float(s._sim.current_time-start)
        else:substeps['time'].append(float(s._sim.current_time-start))
        for key,value in [('q',state.dof_pos),('qd',state.dof_vel),('requested',s.human_requested_torques),('active',s.human_active_torques),('elastic',s.human_elastic_torques),('damping',s.human_damping_torques),('applied',s.human_applied_torques),('negative_cap',s.human_negative_caps),('positive_cap',s.human_positive_caps)]:
            if preallocated_trace:substeps[key][trace_cursor]=value[0].detach().cpu().numpy()
            else:substeps[key].append(value[0].detach().clone())
        trace_cursor+=1
    sim.human_model_trace_callback=record_substep
    writer=imageio.get_writer(str(raw),fps=30,codec='libx264',pixelformat='yuv420p',macro_block_size=2,ffmpeg_params=['-movflags','+faststart']) if render else None
    for frame in range(start_frame,start_frame+frames):
        t=frame/30
        if population_joint:
            chapter=next(c for c in chapters if round(c['start_s']*30)<=frame<round((c['start_s']+c['duration_s'])*30))
            if frame==round(chapter['start_s']*30):
                print('ROM_CHAPTER',chapter['label'],'FRAME',frame,flush=True)
                fixture_pose_common=population_base_pose_common.clone()
                for name,angle in chapter.get('pose_deg',{}).items():fixture_pose_common[:,model.names.index(name)]=math.radians(angle)
                fixture_pose_common[:,model.names.index(chapter['joint'])]=math.radians(chapter['start_deg'])
                root_position=chapter.get('root_pos',protocol['root_pos'])
                if protocol.get('support_mode')=='joint_clamp':
                    # Explicit external rig: narrow engine constraints on OTHER
                    # coordinates. Never narrow the tested coordinate's ROM.
                    nominal=torch.stack((model.lower,model.upper),-1).unsqueeze(0)
                    limits=nominal.clone();half_width=math.radians(.005)
                    limits[:,:,0]=torch.maximum(limits[:,:,0],fixture_pose_common-half_width)
                    limits[:,:,1]=torch.minimum(limits[:,:,1],fixture_pose_common+half_width)
                    test_index=model.names.index(chapter['joint']);limits[:,test_index]=nominal[:,test_index]
                    native_limits=torch.empty_like(limits);native_limits[:,order]=limits
                    view.set_dof_limits(native_limits.cpu(),torch.arange(sim.num_envs,dtype=torch.int32))
                    observed=view.get_dof_limits()[:,order].to(device)
                    if not torch.allclose(observed[:,test_index],nominal[:,test_index],atol=1e-6,rtol=0):
                        raise ValueError('External clamp changed the tested coordinate ROM')
                    clamp_limit_audit.append(observed[0].cpu().numpy().copy())
                # Keep the drawing of the fixed-pelvis rig at the actual reset
                # height when protocols use different limb clearances.
                x,y,z=root_position;seat_top=z-.0843
                for label,position,size in [('seat',[x-.06,y,seat_top-.025],[.34,.38,.05]),
                    ('post',[x-.06,y,(seat_top-.05)/2],[.08,.08,seat_top-.05]),
                    ('base',[x-.06,y,.015],[.55,.55,.03])]:
                    cube=UsdGeom.Cube.Get(sim._sim.stage,'/World/dynamometer_'+label)
                    UsdGeom.XformCommonAPI(cube).SetTranslate(Gf.Vec3d(*position))
                    UsdGeom.XformCommonAPI(cube).SetScale(Gf.Vec3f(*size))
                sim.reset_envs(ResetState(root_pos=torch.tensor([root_position],device=device),
                    root_rot=torch.tensor([chapter.get('root_rot_xyzw',protocol.get('root_rot_xyzw',[0,0,0,1]))],device=device),
                    root_vel=torch.zeros((1,3),device=device),root_ang_vel=torch.zeros((1,3),device=device),
                    dof_pos=fixture_pose_common.clone(),dof_vel=fixture_pose_common*0,state_conversion=StateConversion.COMMON))
        if frame==420 and chapters is None and not population_load and not suspended_gait:
            sim.reset_envs(initial)
            reset_state=sim.get_dof_state()
            reset_checks.append(dict(frame=frame,q_error_rad=float((reset_state.dof_pos-initial.dof_pos).abs().max()),qd_error_rad_s=float((reset_state.dof_vel-initial.dof_vel).abs().max())))
        state=sim.get_dof_state();target=torch.zeros_like(state.dof_pos)
        phase='hold'
        if 2<=t<8:
            phase='joint sweep';target[:,k]=math.radians(45*(1-math.cos(2*math.pi*(t-2)/6)));target[:,w]=math.radians(25*math.sin(2*math.pi*(t-2)/3))
        elif 8<=t<12:phase='passive release'
        elif 12<=t<14:phase='torque saturation'
        elif t>=14:phase='reset and hold'
        if suspended_gait:phase='Synthetic alternating limb targets'
        if population_load:phase='Supine contact settling' if t<4 else 'Supported load equilibrium'
        if chapters is not None:
            ci=next(i for i,c in enumerate(chapters) if round(c['start_s']*30)<=frame<round((c['start_s']+c['duration_s'])*30)) if population_joint else frame//240
            chapter=chapters[ci];phase=chapter['label'];focus=model.names.index(chapter['joint'])
            if population_joint:target=fixture_pose_common.clone()
            if render and (frame==round(chapter['start_s']*30)):
                body=chapter['joint'].rsplit('_',1)[0]
                center=sim.get_bodies_state().rigid_body_pos[0,sim.robot_config.kinematic_info.body_names.index(body)].cpu().numpy()
                close_joint=body in ('R_Ankle','R_Toe','R_Wrist','L_Ankle','L_Toe','L_Wrist')
                offset=np.array([.8,-1.0,.5]) if close_joint else np.array([2.0,-2.0,1.0])
                pv.set_camera_view((center+offset).tolist(),center.tolist())
        # Test-fixture feedback is refreshed at the physical rate. Full coupled
        # inertia avoids applying diagonal-mass gains to low-inertia eigenmodes.
        for substep in range(sim.decimation):
            state=sim.get_dof_state()
            mass=sim._robot.root_physx_view.get_generalized_mass_matrices()[:,base_offset:,base_offset:][:,order][:,:,order]
            mass=mass+torch.diag_embed(fixture_armature)
            if not population_joint:
                acceleration=64*(target-state.dof_pos)-16*state.dof_vel
            if chapters is not None:
                local=t+substep/physics_fps-chapter['start_s']
                if population_joint:
                    # Configured C2 ramp; fixture acts externally, outside human caps.
                    desired,velocity,feedforward=population_fixture_target(chapter,local)
                    target=fixture_pose_common.clone()
                    for name,angle in chapter.get('pose_deg',{}).items():target[:,model.names.index(name)]=math.radians(angle)
                else:
                    desired,velocity,feedforward=joint_demo_target(chapter,local)
                    target.zero_()
                target[:,focus]=desired
                if not population_joint:
                    acceleration=64*(target-state.dof_pos)-16*state.dof_vel
                    acceleration[:,focus]+=16*velocity+feedforward
            if not population_joint:
                command=(mass @ acceleration.unsqueeze(-1)).squeeze(-1)
                command-=model.elastic_torque(state.dof_pos)+model.damping_torque(state.dof_vel)
                if phase=='passive release':command[:,[k,w]]=0
                if phase=='torque saturation':command[:,k]=300 if t<13 else -300
                if population_load:command.zero_()
            if population_joint:
                # No cancellation of plant elastic/active torque: deflection of
                # the external servo is what produces the dynamometer reading.
                fixture_k=chapter.get('fixture_acceleration_gain',protocol.get('fixture_acceleration_gain',1600.))
                fixture_d=2*math.sqrt(fixture_k)
                acceleration=fixture_k*(target-state.dof_pos)-fixture_d*state.dof_vel
                acceleration[:,focus]+=fixture_d*velocity+feedforward
                if chapter.get('rom_suite'):
                    # The tested coordinate is free of the rig. Its unreachable
                    # beyond-limit target must not drive cross-axis support loads.
                    acceleration[:,focus]=0
                gravity=view.get_gravity_compensation_forces()[:,order]
                # This is telemetry only; the commanded fixture does not use C.
                if substep==sim.decimation-1:
                    coriolis=view.get_coriolis_and_centrifugal_compensation_forces()[:,order]
                fixture=(mass@acceleration.unsqueeze(-1)).squeeze(-1)+gravity
                if protocol.get('support_mode')=='joint_clamp':fixture.zero_()
                command=torch.zeros_like(target)
                if chapter['kind']=='isometric':
                    sign=chapter['direction']
                    cap=model.positive[focus] if sign>0 else model.negative[focus]
                    level=min(1.,max(0.,(local-2)/2))
                    command[:,focus]=sign*cap*level
                elif chapter['kind']=='active_rom':
                    command,fixture=active_rom_drive(chapter,focus,state.dof_pos,state.dof_vel,
                        desired,velocity,feedforward,mass,gravity,fixture)
                    if chapter.get('external_probe_nm',0):
                        a,b=chapter['probe_window_s'];a-=.5
                        envelope=max(0.,min(1.,(local-a)/.25,(b-local)/.25))
                        fixture[:,focus]=chapter['direction']*chapter['external_probe_nm']*envelope
                    for key,value in [('time',float(sim._sim.current_time-start)),('chapter_index',ci),
                        ('focus',focus),('target_rad',desired),('fixture_focus_nm',float(fixture[0,focus]))]:
                        active_audit[key].append(value)
            if suspended_gait:
                ref,ref_vel,ref_acc=suspended_gait_target(model.names,t+substep/physics_fps,
                    duration=protocol['duration_s'],ramp=protocol['ramp_s'],frequency=protocol['frequency_hz'])
                target=torch.as_tensor(ref,device=device,dtype=state.dof_pos.dtype).unsqueeze(0)
                velocity=torch.as_tensor(ref_vel,device=device,dtype=state.dof_pos.dtype).unsqueeze(0)
                feedforward=torch.as_tensor(ref_acc,device=device,dtype=state.dof_pos.dtype).unsqueeze(0)
                gain=protocol['acceleration_gain']
                acceleration=gain*(target-state.dof_pos)+2*math.sqrt(gain)*(velocity-state.dof_vel)+feedforward
                command=(mass@acceleration.unsqueeze(-1)).squeeze(-1)
                command+=view.get_gravity_compensation_forces()[:,order]
                command+=view.get_coriolis_and_centrifugal_compensation_forces()[:,order]
                command-=model.elastic_torque(state.dof_pos)+model.damping_torque(state.dof_vel)
            sim._common_actions=command
            sim._apply_control();sim._scene.write_data_to_sim()
            if population_joint:
                total=sim.human_applied_torques+fixture
                native_total=torch.empty_like(total);native_total[:,order]=total
                view.set_dof_actuation_forces(native_total,torch.arange(sim.num_envs,device=device,dtype=torch.int32))
            sim._sim.step(render=False)
            sim._scene.update(dt=sim._sim.get_physics_dt())
        if render:
            sim._sim.render()
            rgb=pv.get_rgb()
            if rgb is None:raise RuntimeError('Missing synchronized rendered frame')
            writer.append_data(rgb)
        after=sim.get_dof_state();data['time'].append(float(sim._sim.current_time-start));data['phase'].append(phase)
        if population_load or population_joint or suspended_gait:
            data['contact_body_forces_w'].append(sim.get_bodies_contact_buf().rigid_body_contact_forces[0].cpu().numpy().copy())
            minimum=0.;contact_count=0
            for sensor in sim._contact_sensor_map.values():
                _,_,_,separations,counts,starts=sensor.contact_physx_view.get_contact_data(1/physics_fps)
                for count,begin in zip(counts.flatten().tolist(),starts.flatten().tolist()):
                    if count:
                        minimum=min(minimum,float(separations[begin:begin+count].min()))
                        contact_count+=count
            data['contact_min_separation_m'].append(minimum);data['contact_points_count'].append(contact_count)
        if population_load:
            body_mass=view.get_masses().to(device);total_mass=body_mass.sum(dim=1,keepdim=True)
            for key,value in [
                ('contact_force_w',sim.get_bodies_contact_buf().rigid_body_contact_forces.sum(dim=1)),
                ('com_w',(sim._robot.data.body_com_pos_w*body_mass.unsqueeze(-1)).sum(dim=1)/total_mass),
                ('com_velocity_w',(sim._robot.data.body_com_lin_vel_w*body_mass.unsqueeze(-1)).sum(dim=1)/total_mass),
                ('root_pos',sim.get_root_state().root_pos),
                ('incoming_joint_wrench',view.get_link_incoming_joint_force())]:
                data[key].append(value[0].detach().cpu().numpy().copy())
        if population_joint:
            data['chapter_index'].append(ci)
            for key,value in [('fixture',fixture),('engine_total',view.get_dof_actuation_forces()[:,order]),
                ('gravity',gravity),('coriolis',coriolis),
                ('inertia_accel',(mass@((after.dof_vel-state.dof_vel)*physics_fps).unsqueeze(-1)).squeeze(-1)),
                ('projected_joint_force',view.get_dof_projected_joint_forces()[:,order]),
                ('contact_force_w',sim.get_bodies_contact_buf().rigid_body_contact_forces.sum(dim=1))]:
                data[key].append(value[0].detach().cpu().numpy().copy())
        if suspended_gait:
            data['root_pos'].append(sim.get_root_state().root_pos[0].cpu().numpy().copy())
            data['native_effort'].append(view.get_dof_actuation_forces()[0,order].cpu().numpy().copy())
            data['gravity'].append(view.get_gravity_compensation_forces()[0,order].cpu().numpy().copy())
            data['coriolis'].append(view.get_coriolis_and_centrifugal_compensation_forces()[0,order].cpu().numpy().copy())
        data['target'].append(target[0].detach().cpu().numpy().copy())
        for key,value in [('q',after.dof_pos),('qd',after.dof_vel),('requested',sim.human_requested_torques),('active',sim.human_active_torques),('elastic',sim.human_elastic_torques),('damping',sim.human_damping_torques),('applied',sim.human_applied_torques),('engine',sim._robot.data.applied_torque[:,sim.data_conversion.dof_convert_to_common])]:
            data[key].append(value[0].detach().cpu().numpy().copy())
        if frame%60==0:print('VIDEO_FRAME',frame,flush=True)
    if writer is not None:writer.close()
    if clamp_limit_audit:
        np.savez_compressed(raw.with_name(raw.stem+'_fixture_limits.npz'),limits_common_rad=np.asarray(clamp_limit_audit),
            scope='External rig limits; tested coordinate retains its original human ROM')
    np.savez_compressed(trace,**{k:np.asarray(v) for k,v in data.items()},dof_names=np.array(model.names),body_names=np.array(sim.robot_config.kinematic_info.body_names),lower=model.lower.cpu().numpy(),upper=model.upper.cpu().numpy(),negative=model.negative.cpu().numpy(),positive=model.positive.cpu().numpy())
    if trace_cursor!=trace_capacity:raise ValueError('Incomplete physical trace buffer')
    step_arrays=substeps if preallocated_trace else {key:np.asarray(value) if key=='time' else torch.stack(value).cpu().numpy() for key,value in substeps.items()}
    np.savez_compressed(raw.with_name(raw.stem+'_substeps.npz'),**step_arrays)
    if active_audit['time']:
        np.savez_compressed(raw.with_name(raw.stem+'_active_audit.npz'),**active_audit)
    metrics=torque_trace_metrics(**step_arrays,lower=model.lower.cpu().numpy(),upper=model.upper.cpu().numpy())
    metrics['terminal_rom_excess_max_deg']=float(torch.rad2deg(torch.maximum(model.lower-after.dof_pos,after.dof_pos-model.upper).clamp_min(0)).max())
    metrics['sampling_grid']=sampling_grid_metrics(step_arrays['time'],expected_dt=1/physics_fps,
        expected_count=frames*(physics_fps//30),terminal_time=data['time'][-1])
    result=dict(status='candidate',raw=str(raw) if render else None,trace=str(trace),frames=frames,fps=30,seed=seed,features=[],physics_fps=480,position_iterations=32,velocity_iterations=8,condition='Fixed pelvis; gravity/self-collision disabled; full-inertia including native armature passive-compensated PD fixture at physics rate except passive release/torque saturation; no trained controller; baseline L0/L1 only',frame_alignment='post-step q/qd and RGB; torques from last physical substep; separate pre-step physics trace',engine_field='IsaacLab actuator applied_torque buffer; not independent PhysX reaction-force measurement',metrics=metrics,shutdown='pending')
    result['model_id']=model_tag
    result['scenario']=scenario;result['chapters']=chapters
    result['physics_device']=physics_device
    result['trace_storage']='preallocated_cpu_arrays' if preallocated_trace else 'tensor_lists'
    if chapters is not None:
        result['condition']='Fixed pelvis; gravity/self-collision off; one joint per 8s chapter; C2 gentle targets with velocity/acceleration feedforward; no forced saturation or release; no trained controller'
    result['reset_checks']=reset_checks
    result['fixed_base']=bool(sim._robot.is_fixed_base)
    result.update(physics_fps=physics_fps,position_iterations=position_iterations,velocity_iterations=velocity_iterations,external_forces_every_iteration=external_forces_every_iteration,start_frame=start_frame,solver_type=solver_type)
    result.update(joint_mode=joint_mode,numerical_frame_mass_kg=frame_mass if joint_mode=='serial' else 0)
    result['kinematic_audit']=kinematic_audit
    if population_joint:
        error=np.rad2deg(np.asarray(data['q'])-np.asarray(data['target']))
        checks=dict(gravity_enabled=not robot.asset.disable_gravity,
            finite=bool(np.isfinite(error).all()),
            fixture_tracking=bool(np.max(np.abs(error))<=2),
            caps=max(metrics['cap_violation_max_nm'])<=2e-4,
            torque_sum=max(metrics['torque_accounting_max_nm'])<=2e-4,
            external_effort_applied=bool(np.max(np.abs(np.asarray(data['engine_total'])-np.asarray(data['applied'])-np.asarray(data['fixture'])))<=.002),
            sampling=bool(metrics['sampling_grid']['passed']))
        checks['contact_penetration']=min(data['contact_min_separation_m'])>=-.005
        checks['rom']=max(metrics['rom_excess_max_deg'])<=math.degrees(1e-4)
        recovered=np.asarray(data['inertia_accel'])+np.asarray(data['coriolis'])+np.asarray(data['gravity'])-np.asarray(data['fixture'])
        selected_closure=[]
        for ci,c in enumerate(chapters):
            mask=np.asarray(data['chapter_index'])==ci;j=model.names.index(c['joint'])
            selected_closure.append(float(np.max(np.abs(recovered[mask,j]-np.asarray(data['applied'])[mask,j]))))
        checks['test_joint_torque_closure']=max(selected_closure)<=.02
        if active_audit['time']:
            checks['active_axis_external_torque_zero']=bool(np.max(np.abs(active_audit['fixture_focus_nm']))==0)
        result['test_joint_torque_closure_max_nm']=selected_closure
        result.update(condition='Fixed pelvis dynamometer fixture; gravity and terrain contacts ON; self-collision OFF; external rotary servo separate from human active/passive torque; no trained controller',
            protocol=protocol,protocol_sha256=sha256(population_config),fixture_tracking_max_deg=float(np.abs(error).max()),
            measurement='Externally applied servo effort, native effort readback and projected joint force recorded; source comparison requires gravity/inertia/contact correction and is not a physical load-cell measurement')
        if protocol.get('profile_path'):
            snapshot=raw.with_name(raw.stem+'_profile.json')
            result['profile_snapshot']=dict(path=str(snapshot),sha256=sha256(snapshot))
        if active_audit['time']:
            result['condition']='Fixed pelvis, gravity/terrain ON, self-collision OFF; test axis external torque exactly zero; torque PD + diagonal inertia feedforward + gravity through human caps; other axes held by external rig; no trained controller'
            result['active_rom_audit']=dict(path=str(raw.with_name(raw.stem+'_active_audit.npz')),
                samples=len(active_audit['time']),external_torque_max_nm=float(np.max(np.abs(active_audit['fixture_focus_nm']))),
                scope='Engineering reachability under specified support/controller; no active-ROM human cohort certification')
        result['engine_gate']=dict(status='passed' if all(checks.values()) else 'candidate_failed_gate',checks=checks)
    elif suspended_gait:
        error=np.rad2deg(np.asarray(data['q'])-np.asarray(data['target']))
        peak=float(np.abs(error).max())
        checks=dict(finite=bool(np.isfinite(error).all()),tracking=peak<=2.,
            caps=max(metrics['cap_violation_max_nm'])<=2e-4,
            torque_sum=max(metrics['torque_accounting_max_nm'])<=2e-4,
            rom=max(metrics['rom_excess_max_deg'])<=.01,
            sampling=bool(metrics['sampling_grid']['passed']),
            fixed_pelvis=bool(np.abs(np.asarray(data['root_pos'])-[3.,3.,1.5]).max()<1e-5),
            no_external_joint_effort=bool(np.abs(np.asarray(data['native_effort'])-np.asarray(data['applied'])).max()<=2e-4),
            gravity_enabled=not robot.asset.disable_gravity)
        result.update(condition='Fixed elevated pelvis; gravity and terrain collision ON; feet airborne; self-collision OFF; synthetic q_ref; full mass-matrix PD/feedforward with gravity, Coriolis and passive compensation, all through human active caps; no policy or external joint servo',
            protocol=protocol,tracking_max_deg=peak,tracking_rms_deg=float(np.sqrt(np.mean(error**2))),
            tracking_per_axis_max_deg=dict(zip(model.names,np.abs(error).max(axis=0).tolist())),
            scope='Multi-joint drive/ROM/torque accounting check only; not gait, balance or human trajectory validation')
        result['engine_gate']=dict(status='passed' if all(checks.values()) else 'candidate_failed_gate',checks=checks)
    elif population_load:
        settled=np.asarray(data['time'])>=4
        forces=np.asarray(data['contact_force_w'])[settled]
        velocities=np.asarray(data['com_velocity_w'])[settled]
        weight=float(total_mass[0,0])*9.81
        checks=dict(free_root=not result['fixed_base'],gravity_enabled=not robot.asset.disable_gravity,
            measured_contact=bool(len(forces) and np.mean(forces[:,2])>0),
            finite=bool(np.isfinite(np.asarray(data['q'])).all()),
            equilibrium=bool(len(forces) and abs(np.mean(forces[:,2])/weight-1)<=.03),
            settled_speed=bool(len(velocities) and np.max(np.linalg.norm(velocities,axis=1))<=.03),
            sampling=bool(metrics['sampling_grid']['passed']))
        checks.update(contact_penetration=min(data['contact_min_separation_m'])>=-.005,
            rom=max(metrics['rom_excess_max_deg'])<=math.degrees(1e-4),
            caps=max(metrics['cap_violation_max_nm'])<=2e-4,
            torque_sum=max(metrics['torque_accounting_max_nm'])<=2e-4)
        result.update(condition='Free-root supine body on terrain; gravity and terrain contact ON; self-collision OFF; zero active torque; no controller or hidden root support',
            mass_kg=float(total_mass[0,0]),gravity_m_s2=9.81,
            support_mean_n=None if not len(forces) else float(forces[:,2].mean()),
            population_reference='Static force balance m*g is a mechanical reference, not a human population mean/SD')
        result['engine_gate']=dict(status='passed' if all(checks.values()) else 'candidate_failed_gate',checks=checks)
    elif chapters is None:
        result['engine_gate']=video_engine_checks(data,metrics,reset_checks,result['fixed_base'],physics_fps)
    else:
        tracking=np.abs(np.asarray(data['q'])-np.asarray(data['target']))
        checks=dict(rom=max(metrics['rom_excess_max_deg'])<=math.degrees(1e-4),
            caps=max(metrics['cap_violation_max_nm'])<=2e-4,
            torque_sum=max(metrics['torque_accounting_max_nm'])<=2e-4,
            finite=bool(np.isfinite(np.asarray(data['q'])).all() and np.isfinite(np.asarray(data['qd'])).all()),
            tracking=bool(tracking.max()<=math.radians(2)),
            speed=bool(np.abs(np.asarray(data['qd'])).max()<=math.radians(60)),
            sampling=bool(metrics['sampling_grid']['passed']),
            chapters=bool(np.array_equal(data['phase'],np.repeat([c['label'] for c in chapters],240))),
            actuator_buffer=bool(np.abs(np.asarray(data['engine'])-np.asarray(data['applied'])).max()<=2e-4))
        result['demo_tracking_max_deg']=float(np.rad2deg(tracking.max()))
        result['demo_speed_max_deg_s']=float(np.rad2deg(np.abs(np.asarray(data['qd'])).max()))
        result['engine_gate']=dict(status='passed' if all(checks.values()) else 'candidate_failed_gate',checks=checks,
            scope='Gentle joint chapters; tracking <=2deg, speed <=60deg/s; literature bounds are source adoption, not biological trajectory validation')
    if population_joint and not active_audit['time']:
        from .metrics import population_assay_metrics,population_band_metrics
        try:
            result['population_assays']=population_assay_metrics(np.load(trace),chapters)
            predicted=[];means=[];sds=[]
            for row in result['population_assays']:
                if 'predicted' in row:
                    predicted.extend(row['predicted']);means.extend(np.atleast_1d(row['reference']['mean']).tolist());sds.extend(np.atleast_1d(row['reference']['sd']).tolist())
            score=population_band_metrics(predicted,means,sds)
            result['population_gate']=dict(status='passed_descriptive_scale' if score['rms_z']<=1 and score['within_2sd_fraction']>=.95 else 'candidate_failed_gate',metrics=score,
                scope='Aggregate predeclared assay points; inspect each source/protocol limitation; not independent biological validation')
        except ValueError as error:
            result['population_gate']=dict(status='candidate_failed_gate',reason=str(error))
    if population_joint and protocol.get('assay')=='all_direction_active_rom':
        from .metrics import all_direction_rom_metrics
        result['legacy_fixture_gate']=result['engine_gate']
        rows=all_direction_rom_metrics(np.load(trace),step_arrays,active_audit,chapters,protocol['engineering_thresholds'])
        result['direction_results']=rows
        result['engine_gate']={'status':'passed' if all(r['status']=='passed' for r in rows) else 'candidate_failed_gate',
            'scope':'All-direction engineering assay, not independent biological certification',
            'checks':{'direction_gates':all(r['status']=='passed' for r in rows),'sampling':bool(metrics['sampling_grid']['passed'])}}
        if not metrics['sampling_grid']['passed']:result['engine_gate']['status']='candidate_failed_gate'
        result['condition']=protocol['scope']+'; zero-strength axes use labelled external 0.5 Nm limit probes, not active ROM'
    result_path=raw.with_suffix('.json');result_path.write_text(json.dumps(result,indent=2))
    print('VIDEO_CAPTURE_SAVED',flush=True)
    if pv is not None:
        pv.rgb_annotator.detach([pv.render_product.path]);pv.render_product.destroy()
    # clear_instance unsubscribes the STOP callback which otherwise renders
    # forever while the timeline is stopped. Do not call stop() with it attached.
    sim._sim.clear_all_callbacks()
    from isaaclab.sim import SimulationContext
    SimulationContext.clear_instance()
    sim.human_model_trace_callback=None
    del sim,model,mass,inertia,root,dof,initial,state,after,target,command,value,pv
    import gc
    gc.collect();torch.cuda.empty_cache()
    print('VIDEO_SHUTDOWN_BEGIN',flush=True)
    app.close(wait_for_replicator=False)
    result['shutdown']='returned';result_path.write_text(json.dumps(result,indent=2))
    print('VIDEO_COMPLETE',raw,flush=True)


def run(output, stage, seed, features, backend='mujoco'):
    started = time.time()
    output = Path(output).resolve()
    label = f"healthy_adult_v1_{'_'.join(features) or 'baseline'}_seed{seed}"
    if backend!='mujoco':
        label+=f'_{backend}'
    path = output/"evaluation"/stage/f"{label}.json"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing evidence: {path}")
    path.parent.mkdir(parents=True,exist_ok=True)
    logs = output/"logs"; logs.mkdir(parents=True,exist_ok=True)
    profile = load_profile()
    model = HumanJointModel(profile,list(profile["joints"]),dtype=torch.float64,features=features)
    resources = {"before":resource_snapshot()}
    result = {"implementation_id":model.profile_id,"features":features,"seed":seed,"stage":stage,
              "status":"candidate_failed_gate", "reason":"Independent joint-level biological validation incomplete",
              "contract_sha256":sha256(v1_resource("validation_contract.json")),
              "candidate_parameters_sha256":sha256(v1_resource("candidate_parameters.json")),
              "source_sha256":{f.name:sha256(f) for f in Path(__file__).parent.glob('*.py')},
              "numerical":numerical_assays(model,seed)}
    result["strength"], samples = strength_assays(model)
    with path.with_suffix('.csv').open('w') as f:
        writer=csv.writer(f); writer.writerow(['side','direction','clinical_angle_rad','concentric_velocity_rad_s','predicted_nm','source_equation_nm']); writer.writerows(samples)
    result["activation"] = activation_assays(model)
    result['wrist_damping'] = wrist_damping_assay(model)
    try:
        with (logs/f"{label}_{backend}.log").open('w') as log, contextlib.redirect_stdout(log):
            result[backend] = (isaaclab_assay if backend=='isaaclab' else mujoco_assay)(features,path.with_suffix('.npz'))
    except BaseException as error:
        result['reason']='Backend assay did not complete'
        result['backend_error']={'type':type(error).__name__,'message':str(error)}
        path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        raise
    finally:
        resources['after']=resource_snapshot()
        (logs/f"{label}_resources.json").write_text(json.dumps(resources,indent=2))
    contract = json.loads(v1_resource('validation_contract.json').read_text())
    result['evaluation_requirements'] = evaluation_requirements(contract)
    result['elapsed_s']=time.time()-started
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'path':str(path),'status':result['status'],'elapsed_s':result['elapsed_s']}))


def passive_candidate_selection(candidates, threshold=.15):
    """Compare identical screening supports; no mean-only adoption decision."""
    rows=[]; support=None; reference_context=None
    for source,result in candidates:
        current=sorted((fold['held_out_study'],curve['id'],curve['candidate']['n'])
                       for fold in result['folds'] for curve in fold['curves'])
        if support is not None and current!=support:
            return {'status':'incomparable','reason':'Different held-out curve/sample support; no ranking'}
        support=current
        context=(result.get('source_sha256'),result.get('split_sha256'),
                 result.get('fitted',{}).get('calibration_scales_nm'))
        if reference_context is not None and context!=reference_context:
            return {'status':'incomparable','reason':'Different source, split or normalization; no ranking'}
        reference_context=context
        errors=[curve['candidate']['nrmse'] for fold in result['folds'] for curve in fold['curves']]
        rows.append({'source':source,'slope_scale':result.get('slope_scale',1.),
                     'study_macro_nrmse':result['candidate_study_macro_nrmse'],
                     'worst_curve_nrmse':max(errors),'failed_curves':sum(e>threshold for e in errors),
                     'experiment_gate_status':'candidate_failed_gate' if any(e>threshold for e in errors) else 'screening_targets_only'})
    keys=('study_macro_nrmse','worst_curve_nrmse','failed_curves')
    for row in rows:
        dominators=[other['source'] for other in rows
                    if all(other[k]<=row[k] for k in keys) and any(other[k]<row[k] for k in keys)]
        row.update(selection_status='archived' if dominators else 'candidate',dominated_by=dominators)
    return {'status':'screening_only','runtime_adoption':False,
            'reason':'Compare mean, worst curve and failure count together; independent confirmation still required',
            'curve_threshold':threshold,'candidates':rows}


def summarize(output):
    """Refresh the experiment's machine-readable evidence index, not adoption."""
    output=Path(output).resolve()
    rows=[]; sources=[]; passive_candidates=[]
    provenance_fields = ('metric_id', 'threshold_basis', 'contract_sha256',
                         'contract_required_evidence', 'assay_required_evidence',
                         'contract_evidence_satisfied', 'required_evidence')
    def visit(value, prefix, source):
        if isinstance(value, dict):
            if 'status' in value and 'threshold' in value and 'value' in value:
                rows.append({'source':source,'metric_path':prefix,
                             'status':value['status'],'value':value['value'],
                             'threshold':value['threshold'],
                             'evidence_kind':value.get('evidence_kind','unspecified'),
                             **{key: value.get(key) for key in provenance_fields}})
            for key, child in value.items():
                visit(child, f'{prefix}.{key}' if prefix else key, source)
        elif isinstance(value, list):
            for i, child in enumerate(value):
                visit(child,f'{prefix}[{i}]',source)
    for path in sorted((output/'evaluation').rglob('*.json')):
        source=str(path.relative_to(output))
        result=json.loads(path.read_text())
        if path.name.startswith('healthy_adult_v1_passive_study_fit') and 'folds' in result:
            passive_candidates.append((source,result))
        sources.append({'path':source,'sha256':sha256(path),
                        'status':result.get('status','unspecified'),
                        'scope':result.get('scope',result.get('reason','see source'))})
        visit(result,'',source)
    deliverables=output/'deliverables'; deliverables.mkdir(exist_ok=True)
    with (deliverables/'metrics_summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['source','metric_path','status','value','threshold','evidence_kind',
                                          *provenance_fields])
        writer.writeheader();writer.writerows(rows)
    contract_path = v1_resource('validation_contract.json')
    contract = json.loads(contract_path.read_text())
    summary={'implementation_id':'healthy_adult_v1','status':'candidate_failed_gate',
             'reason':'Independent joint-level biological validation incomplete; passive screening has failed curves.',
             'active_scope':'joint_level_plant',
             'scope_policy':contract['scope_policy'],
             'contract_sha256':sha256(contract_path),
             'evaluation_requirements':evaluation_requirements(contract),
             'passive_candidate_selection':passive_candidate_selection(passive_candidates,
                 next(r['threshold_max'] for r in contract['metrics'] if r['id']=='passive_curve_nrmse')),
             'scope':'Interim evidence inventory; each metric retains its assay conditions. Counts are not contract completion.',
             'sources':sources,'recorded_gate_count':len(rows)}
    (deliverables/'result.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    manifest={'implementation_id':'healthy_adult_v1','status':'candidate_failed_gate',
              'files':[{'path':str(p.relative_to(output)),'sha256':sha256(p),'bytes':p.stat().st_size}
                       for p in sorted(output.rglob('*')) if p.is_file() and p!=output/'manifest.json']}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'sources':len(sources),'recorded_gates':len(rows),'manifest_files':len(manifest['files'])}))


def finalize_all_direction_videos(output, sources_path):
    """Review explicitly selected native runs and publish four trace-linked MP4s.

    Raw reports remain immutable. A separate review preserves nominal-clock
    discrepancies and records the native float32 callback grid assessment.
    """
    import shutil
    import zipfile
    from .metrics import all_direction_rom_metrics,native_clock_grid_metrics
    from human_controller.visualization.video_compositor import compose_active_rom_video
    output=Path(output).resolve();sources=json.loads(Path(sources_path).read_text())
    if set(sources)!=set(('lower','feet','upper','axial')):raise ValueError('Four explicit groups required')
    root=WORKSPACE_ROOT;deliverables=output/'deliverables';videos=root/'videos'
    reviews={};all_rows=[];coverage=set()
    for group,source in sources.items():
        path=Path(source);path=path if path.is_absolute() else output/path
        report=json.loads(path.read_text())
        if report['shutdown']!='process_exit_0':raise ValueError(f'Native process not complete: {path}')
        if report['protocol']['assay']!='all_direction_active_rom':raise ValueError('Unexpected assay')
        trace=Path(report['trace']);raw=Path(report['raw'])
        with np.load(trace) as df,np.load(raw.with_name(raw.stem+'_substeps.npz')) as sf,np.load(raw.with_name(raw.stem+'_active_audit.npz')) as af:
            d={k:df[k] for k in df.files};s={k:sf[k] for k in sf.files};a={k:af[k] for k in af.files}
            rows=all_direction_rom_metrics(d,s,a,report['chapters'],report['protocol']['engineering_thresholds'])
            clock=native_clock_grid_metrics(s['time'],physics_fps=report['physics_fps'],expected_count=report['frames']*(report['physics_fps']//30),terminal_time=float(d['time'][-1]))
            limits=np.load(raw.with_name(raw.stem+'_fixture_limits.npz'))['limits_common_rad']
            if len(limits)!=len(rows):raise ValueError('Missing external fixture limit readback')
            names=list(d['dof_names']);nominal=np.stack([d['lower'],d['upper']],-1)
            fixture_ok=True
            for i,c in enumerate(report['chapters']):
                j=names.index(c['joint']);other=np.arange(len(names))!=j
                fixture_ok &= bool(np.allclose(limits[i,j],nominal[j],atol=1e-6,rtol=0))
                fixture_ok &= bool(np.rad2deg(limits[i,other,1]-limits[i,other,0]).max()<=.0101)
            peak_all=float(np.rad2deg(np.abs(s['qd'])).max())
        for row in rows:
            key=(row['joint'],row['direction'])
            if key in coverage:raise ValueError(f'Duplicate direction {key}')
            coverage.add(key);all_rows.append(dict(group=group,seed=report['seed'],**row))
        legacy=report['legacy_fixture_gate']['checks']
        checks={k:legacy[k] for k in ('gravity_enabled','finite','caps','torque_sum','external_effort_applied','contact_penetration')}
        checks.update(direction_gates=all(r['status']=='passed' for r in rows),native_clock_grid=clock['passed'],
            fixture_limit_readback=fixture_ok,fixed_pelvis=report['fixed_base'])
        review=dict(source_report=str(path),source_report_sha256=sha256(path),trace_sha256=sha256(trace),
            original_engine_gate=report['engine_gate'],sampling_clock=clock,direction_results=rows,
            all_coordinate_peak_speed_deg_s=peak_all,
            delivery_gate=dict(status='passed' if all(checks.values()) else 'candidate_failed_gate',checks=checks),
            scope='Engineering test of original tested-axis ROM with explicit external non-test clamps; no independent human AROM certification')
        name=f'healthy_adult_v1_active_rom_{group}_gravity_contact_seed{report["seed"]}'
        (deliverables/f'{name}_experiment_validation.json').write_text(json.dumps(review,indent=2,allow_nan=False))
        reviews[group]=(report,raw,trace,review,name)
    expected={(n,d) for n in load_profile()['joints'] for d in (-1,1)}
    if coverage!=expected:raise ValueError('Missing or unexpected directions in delivery')
    results=[]
    for group,(report,raw,trace,review,name) in reviews.items():
        destination=deliverables/(name+'.mp4');validation_path=deliverables/(name+'_video_validation.json')
        if destination.exists():
            validation=json.loads(validation_path.read_text())
            if validation['video_sha256']!=sha256(destination) or validation['trace_sha256']!=sha256(trace):
                raise ValueError('Existing video evidence mismatch; do not overwrite')
        else:
            validation=compose_active_rom_video(raw,trace,destination,review=review)
            validation_path.write_text(json.dumps(validation,indent=2))
        copy=videos/destination.name
        if copy.exists() and sha256(copy)!=sha256(destination):
            copy=videos/f'{destination.stem}_{time.strftime("%Y%m%d_%H%M%S")}.mp4'
        if not copy.exists():shutil.copy2(destination,copy)
        if sha256(copy)!=sha256(destination):raise ValueError('Video copy SHA mismatch')
        results.append(dict(group=group,path=str(copy.relative_to(root)),source=str(destination.relative_to(root)),
            sha256=sha256(destination),video_validation=validation,experiment_gate_status=review['delivery_gate']['status'],
            cases=len(review['direction_results']),passed=sum(r['status']=='passed' for r in review['direction_results'])))
        print('ROM_VIDEO_DELIVERED',copy,results[-1]['experiment_gate_status'],flush=True)
    with (deliverables/'metrics_summary.csv').open('w') as f:
        keys=[k for k in all_rows[0] if k not in ('checks','residual_scope')]+['failed_checks']
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for row in all_rows:writer.writerow({**{k:row[k] for k in keys if k!='failed_checks'},'failed_checks':','.join(k for k,v in row['checks'].items() if not v)})
    status='passed_engineering_scope' if all(r['experiment_gate_status']=='passed' for r in results) else 'candidate_failed_gate'
    result=dict(implementation_id='healthy_adult_v1',status=status,scope='All 138 coordinate directions; active reachability and zero-strength constraint probes are separate; not biological certification',
        conditions='CPU PhysX 480Hz, gravity/terrain ON, elevated limbs, self-collision OFF; other coordinates externally clamped; tested ROM unchanged',
        direction_count=len(all_rows),active_direction_count=sum(not r['locked_axis'] for r in all_rows),
        locked_direction_count=sum(r['locked_axis'] for r in all_rows),passed=sum(r['status']=='passed' for r in all_rows),
        failed=[dict(joint=r['joint'],direction=r['direction'],checks=[k for k,v in r['checks'].items() if not v]) for r in all_rows if r['status']!='passed'],videos=results)
    (deliverables/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    index_path=videos/'index.json';index=json.loads(index_path.read_text())
    for r in results:
        if any(e['path']==r['path'] for e in index['entries']):continue
        index['entries'].append(dict(name=Path(r['path']).name,path=r['path'],source=r['source'],sha256=r['sha256'],source_sha256=r['sha256'],
            user_selection='pending_feedback',selection_status='pending_feedback',retention_adopted=False,
            video_validation=r['video_validation'],experiment_gate_status=r['experiment_gate_status']))
    index_path.write_text(json.dumps(index,indent=2,ensure_ascii=False)+'\n')
    package=Path(__file__).parent;compositor=root/'src/human_controller/visualization/video_compositor.py'
    with zipfile.ZipFile(deliverables/'analysis_source.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in [Path(__file__),package/'metrics.py',compositor]:z.write(p,p.relative_to(root))
    manifest=dict(implementation_id='healthy_adult_v1',status=status,selection_status='candidate',
        files=[dict(path=str(p.relative_to(output)),sha256=sha256(p),bytes=p.stat().st_size,
                    status='archived' if 'screening' in p.parts or 'scratch' in p.parts else 'candidate')
            for p in sorted(output.rglob('*')) if p.is_file() and p.name!='manifest.json' and not p.is_symlink()])
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('ROM_DELIVERY_RESULT',status,result['passed'],'/',len(all_rows),flush=True)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--stage',choices=['screening','confirmation','ablation'])
    parser.add_argument('--seed',type=int)
    parser.add_argument('--summarize',action='store_true')
    parser.add_argument('--finalize-rom-videos',help='JSON mapping of four groups to explicit native report paths')
    parser.add_argument('--validation-video',action='store_true')
    parser.add_argument('--no-render',action='store_true',help='Run the video assay measurements without rendering')
    parser.add_argument('--video-frames',type=int,default=480)
    parser.add_argument('--video-start-frame',type=int,default=0)
    parser.add_argument('--physics-fps',type=int,default=480)
    parser.add_argument('--position-iterations',type=int,default=32)
    parser.add_argument('--velocity-iterations',type=int,default=1)
    parser.add_argument('--external-forces-every-iteration',action='store_true')
    parser.add_argument('--solver-type',type=int,choices=[0,1],default=1)
    parser.add_argument('--joint-mode',choices=['d6','serial'],default='serial')
    parser.add_argument('--frame-mass',type=float,default=1e-6)
    parser.add_argument('--audit-kinematics',action='store_true')
    parser.add_argument('--scenario',choices=['stress','lower_body','upper_body','population_load','population_joint','suspended_gait','v2_suite'],default='stress')
    parser.add_argument('--population-config')
    parser.add_argument('--video-worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--features',default='')
    parser.add_argument('--backend',choices=['mujoco','isaaclab'],default='mujoco')
    args=parser.parse_args()
    torch.set_num_threads(2)
    if args.validation_video:
        if args.video_worker:
            isaaclab_validation_video(args.output,args.seed or 2400,render=not args.no_render,frames=args.video_frames,physics_fps=args.physics_fps,start_frame=args.video_start_frame,position_iterations=args.position_iterations,velocity_iterations=args.velocity_iterations,external_forces_every_iteration=args.external_forces_every_iteration,solver_type=args.solver_type,joint_mode=args.joint_mode,frame_mass=args.frame_mass,audit_kinematics=args.audit_kinematics,scenario=args.scenario,population_config=args.population_config)
        else:
            # Kit may terminate its interpreter inside close(). An independent
            # parent records the actual exit code instead of claiming close returned.
            import sys
            seed=args.seed or 2400
            output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
            assay_name='v2_suite' if args.scenario=='v2_suite' else 'suspended_gait' if args.scenario=='suspended_gait' else 'population_joint_gravity_contact' if args.scenario=='population_joint' else ('population_load_gravity_contact' if args.scenario=='population_load' else 'l0_l1_fixedbase')
            model_tag=json.loads(Path(args.population_config).read_text()).get('model_id','healthy_adult_v1') if args.population_config else 'healthy_adult_v1'
            lifecycle=output/f'{model_tag}_{assay_name}_seed{seed}_lifecycle.json'
            if lifecycle.exists():raise FileExistsError(lifecycle)
            record={'resources_before':resource_snapshot(),'source_sha256':sha256(__file__),
                    'command':[sys.executable,'-u','-m','protomotions.robot_configs.human_model.validation',*sys.argv[1:],'--video-worker']}
            # Execution evidence only: snapshots are archived, never imported.
            import zipfile
            bundle=output/f'{model_tag}_{assay_name}_seed{seed}_source.zip'
            package_root=PROTOMOTIONS_ROOT
            source_files=list(PACKAGE_ROOT.rglob('*.py'))+list(PACKAGE_ROOT.rglob('*.json'))
            source_files += [package_root/'protomotions'/p for p in ('simulator/isaaclab/simulator.py','simulator/isaaclab/utils/scene.py','simulator/isaaclab/utils/perspective_viewer.py','simulator/base_simulator/simulator.py','robot_configs/smpl.py')]
            source_files += [package_root/'protomotions/data/assets'/p for p in ('usd/smpl_humanoid_healthy_adult_v1.usda','mjcf/smpl_humanoid.xml')]
            with zipfile.ZipFile(bundle,'x',compression=zipfile.ZIP_DEFLATED) as archive:
                for source in source_files:archive.write(source,source.relative_to(package_root))
            record['source_bundle']={'path':str(bundle),'sha256':sha256(bundle),'status':'archived'}
            lifecycle.write_text(json.dumps(record,indent=2))
            child=subprocess.Popen(record['command'])
            record['pid']=child.pid;record['exit_code']=child.wait()
            record['resources_after']=resource_snapshot()
            result_path=output/f'{model_tag}_{assay_name}_seed{seed}_raw.json'
            record['result_saved']=result_path.exists()
            record['supervisor_exit_code']=record['exit_code'] or (0 if record['result_saved'] else 1)
            if result_path.exists():
                result=json.loads(result_path.read_text())
                result['shutdown']='process_exit_0' if record['exit_code']==0 else 'process_failed'
                result['exit_code']=record['exit_code']
                result_path.write_text(json.dumps(result,indent=2))
            lifecycle.write_text(json.dumps(record,indent=2))
            print('VIDEO_WORKER_EXIT',record['pid'],record['exit_code'],flush=True)
            sys.exit(record['supervisor_exit_code'])
    elif args.finalize_rom_videos:
        finalize_all_direction_videos(args.output,args.finalize_rom_videos)
    elif args.summarize:
        summarize(args.output)
    elif args.stage is None or args.seed is None:
        parser.error('--stage and --seed are required for an assay')
    else:
        run(args.output,args.stage,args.seed,tuple(filter(None,args.features.split(','))),args.backend)
