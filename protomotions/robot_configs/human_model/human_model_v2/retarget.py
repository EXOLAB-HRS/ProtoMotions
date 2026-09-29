"""SMPL packed motions to fixed v2 hinges, with explicit quality/provenance gates.

Uses source world transforms rather than assuming SMPL exp-map coordinates are
v2 Euler hinges. Failed conversions remain diagnostic candidates, not AMP input.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from protomotions.components.pose_lib import (extract_kinematic_info,
    extract_transforms_from_qpos_non_root, compute_cartesian_velocity,
    compute_angular_velocity)
from protomotions.utils.rotations import quaternion_to_matrix, matrix_to_quaternion
from ..common.paths import PACKAGE_ROOT


def forward(ki, q, root_pos, root_rot):
    local = extract_transforms_from_qpos_non_root(ki,q,False)
    positions, rotations = [root_pos], [root_rot]
    for b in range(1,ki.num_bodies):
        parent = ki.parent_indices[b]
        rotations.append(rotations[parent] @ ki.local_rot_ref_mat[b] @ local[:,b])
        positions.append(positions[parent] + (rotations[parent] @ ki.local_pos[b,:,None]).squeeze(-1))
    local = torch.cat((root_rot[:,None], local[:,1:]),1)
    return torch.stack(positions,1),torch.stack(rotations,1),local



def refine_upper_trajectory(ki, original_q, root, target, target_rot, position_ids,
                            rotation_ids, dt, cutoff_hz, iterations, arms_only=False, source_ki=None):
    """Source-position constrained temporal fitting; root and leg hinges frozen."""
    from .retarget_metrics import smooth_upper_hinges
    upper=([i for i,n in enumerate(ki.dof_names) if any(part in n for part in ('Shoulder_','Elbow_','Wrist_','Hand_'))] if arms_only else list(range(14,ki.num_dofs)))
    baseline=original_q.detach().clone()
    with torch.no_grad():
        old_pos,_,_=forward(ki,baseline,root,target_rot[:,0])
        start=smooth_upper_hinges(baseline,ki.dof_names,dt,cutoff_hz)
    reference=None
    if source_ki is not None:
        if not arms_only:raise ValueError('Proximal priority requires arms-only fitting')
        from .retarget_metrics import source_arm_reference
        reference=source_arm_reference(target_rot,source_ki,ki,dt)
        start[:,upper]=reference['q'][:,upper]
        angle_scale=reference['angle_scale'][upper]
        rate_scale=reference['rate_scale'][upper]
    param=torch.nn.Parameter(start[:,upper].clone())
    optimizer=torch.optim.Adam([param],lr=.003)
    bodies=[ki.body_names.index(n) for n in ('Chest','L_Shoulder','R_Shoulder')]
    for step in range(iterations):
        optimizer.zero_grad()
        q=baseline.clone();q[:,upper]=param
        pos,rot,_=forward(ki,q,root,target_rot[:,0])
        qjerk=torch.diff(param,n=3,dim=0)/dt**3
        relative=target_rot[:,0,None].transpose(-1,-2)@rot[:,bodies]
        delta=relative[1:]@relative[:-1].transpose(-1,-2)
        omega=torch.stack((delta[...,2,1]-delta[...,1,2],delta[...,0,2]-delta[...,2,0],delta[...,1,0]-delta[...,0,1]),-1)/(2*dt)
        angular_jerk=torch.diff(omega,n=2,dim=0)/dt**2
        if reference is None:
            loss=10*(pos[:,position_ids]-target).square().mean()+.005*(rot[:,rotation_ids]-target_rot).square().mean()
            loss+=1e-7*qjerk.square().mean()+1e-8*angular_jerk.square().mean()
        else:
            residual=param-reference['q'][:,upper]
            # Dimensionless source-angle/rate errors dominate weak distal IK.
            loss=(residual/angle_scale).square().mean()
            loss+=(torch.diff(residual,dim=0)/dt/rate_scale).square().mean()
            loss+=.05*((pos[:,position_ids]-target)/.1).square().mean()
            loss+=1e-8*(torch.diff(residual,n=3,dim=0)/dt**3).square().mean()
        if not arms_only:loss+=100*((pos-old_pos).norm(dim=-1)-.025).clamp_min(0).square().mean()
        loss.backward();optimizer.step()
        with torch.no_grad():
            param.clamp_(ki.dof_limits_lower[upper],ki.dof_limits_upper[upper])
            if not arms_only:
                param[0]=baseline[0,upper];param[-1]=baseline[-1,upper]
        if step in (iterations//2,3*iterations//4):
            optimizer.param_groups[0]['lr']*=.1
    result=baseline.clone();result[:,upper]=param.detach()
    if reference is not None:
        # Clip-wide per-axis residual scaling preserves continuity, phase and ROM.
        residual=result[:,upper]-reference['q'][:,upper]
        scale=torch.minimum(torch.ones_like(angle_scale),angle_scale/residual.abs().amax(0).clamp_min(1e-12))
        scale=torch.minimum(scale,rate_scale/(torch.diff(residual,dim=0)/dt).abs().amax(0).clamp_min(1e-12))
        result[:,upper]=reference['q'][:,upper]+residual*scale
    return result


def project_upper_correction(ki, q, reference_q, root, root_rot, radius_m):
    """One clip-wide blend bounds FK displacement without frame-wise clipping."""
    if radius_m <= 0 or not torch.equal(q[:,:14],reference_q[:,:14]):
        raise ValueError('Upper projection requires positive radius and identical legs')
    with torch.no_grad():
        reference_pos=forward(ki,reference_q,root,root_rot)[0]
        def displacement(alpha):
            trial=reference_q+alpha*(q-reference_q)
            return float((forward(ki,trial,root,root_rot)[0]-reference_pos).norm(dim=-1).max())
        if displacement(1.)<=radius_m:return q.clone(),1.
        lo,hi=0.,1.
        for _ in range(16):
            mid=(lo+hi)/2
            if displacement(mid)<=radius_m:lo=mid
            else:hi=mid
        return reference_q+lo*(q-reference_q),lo


def audit_reference(pack_path, output_path):
    """Screen reference geometry, posture and forward speed coverage without training.

    Contact labels in mocap are proxies, not measured simulated forces. This
    audit never silently upgrades the packed file to a training-approved source.
    """
    import mujoco
    import numpy as np
    from protomotions.envs.control.walking_quality import posture_angles,StepTracker
    pack_path,output_path=Path(pack_path),Path(output_path)
    data=torch.load(pack_path,map_location='cpu',weights_only=False)
    meta=data['human_model_metadata'];names=meta['body_names']
    mj=mujoco.MjModel.from_xml_path(str(PACKAGE_ROOT/'human_model_v2/assets/human_model_v2.xml'))
    feet=[names.index(x) for x in ('L_Ankle','L_Toe','R_Ankle','R_Toe')]
    positions=data['gts'];rot=quaternion_to_matrix(data['grs'],w_last=True)
    bottom=[]
    for b in feet:
        bid=mujoco.mj_name2id(mj,mujoco.mjtObj.mjOBJ_BODY,names[b])
        for g in np.where(mj.geom_bodyid==bid)[0]:
            if mj.geom_type[g]!=mujoco.mjtGeom.mjGEOM_BOX:raise ValueError('Foot geometry audit only supports current v2 boxes')
            corners=torch.tensor([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)],dtype=torch.float32)*torch.tensor(mj.geom_size[g],dtype=torch.float32)
            gr=quaternion_to_matrix(torch.tensor(mj.geom_quat[g],dtype=torch.float32),w_last=False)
            corners=corners@gr.T+torch.tensor(mj.geom_pos[g],dtype=torch.float32)
            world=positions[:,b,None]+corners[None]@rot[:,b].transpose(-1,-2)
            bottom.append(world[:,:,2].min(-1).values)
    lowest=torch.stack(bottom,-1).min(-1).values
    rows=[]
    for mid,n in enumerate(data['motion_num_frames']):
        a=int(data['length_starts'][mid]);n=int(n);dt=float(data['motion_dt'][mid]);sl=slice(a,a+n)
        trunk,head=posture_angles(data['grs'][sl],positions[sl],*[names.index(x) for x in ('Pelvis','Chest','Head')])
        vel=data['gvs'][sl,0,:2];speed=vel.norm(dim=-1);direction=vel/speed[:,None].clamp_min(.01)
        heading=rot[sl,0,:2,0];alignment=(heading*direction).sum(-1)
        track=StepTracker(1,'cpu',dt);cadence=[];length=[]
        for t in range(n):
            contact=torch.stack([data['contacts'][a+t,feet[:2]].any(),data['contacts'][a+t,feet[2:]].any()])[None]
            ok=track.update(contact,positions[a+t,[feet[0],feet[2]],:2][None],direction[t:t+1])
            if ok[0] and speed[t]>.3 and alignment[t]>.85:
                cadence.append(track.cadence[0].clone());length.append(track.step_length[0].clone())
        rows.append(dict(motion_id=mid,frames=n,speed_median_mps=float(speed.median()),forward_fraction=float((alignment>.85).float().mean()),
            cadence_median_spm=float(torch.stack(cadence).median()) if cadence else None,
            step_length_median_m=float(torch.stack(length).median()) if length else None,
            trunk_p95_deg=float(torch.quantile(trunk,.95)),head_pitch_abs_p95_deg=float(torch.quantile(head.abs(),.95)),
            ground_penetration_max_m=float((-lowest[sl]).clamp_min(0).max())))
    coverage={str(v):any(abs(r['speed_median_mps']-v)<=.125 and r['forward_fraction']>=.8 for r in rows) for v in (.5,.75,1.,1.25,1.5)}
    result=dict(source=str(pack_path),source_sha256=hashlib.sha256(pack_path.read_bytes()).hexdigest(),motions=rows,speed_coverage=coverage,
        status='screening_only',scope='Mocap contact proxy; geometry and posture report, not policy gait success; no cohort SD')
    output_path.write_text(json.dumps(result,indent=2)+'\n');return result


def convert(source, destination, *, iterations=800, device='cpu', motion_ids=None, seed=0, initial_pack=None, upper_smoothing_hz=None, upper_refine_iterations=0, trust_reference_pack=None, upper_trust_radius_m=.0295, source_trunk=False, source_trunk_arm_iterations=0, proximal_arms=False):
    """smpl2hm bounded, per-clip optimization; failed clips are never approved.

    Whole-clip fitting keeps boundary continuity. Clips are processed sequentially
    to bound RAM/VRAM; per-clip reports flush immediately for interrupted runs.
    """
    from .retarget_metrics import (FOOT_NAMES, box_corners, world_points,
        contact_targets, material_points, derivative, contact_metrics, gait_metrics, normalize_skeleton)
    source, destination = Path(source), Path(destination)
    if destination.exists() or destination.with_suffix('.json').exists():
        raise FileExistsError(destination)
    if iterations < 0 or (iterations == 0 and initial_pack is None):
        raise ValueError('zero iterations requires a provenance-checked warm pack')
    if proximal_arms and (not source_trunk or not source_trunk_arm_iterations or initial_pack is None):
        raise ValueError('Proximal arms requires source-trunk, positive arm iterations and frozen v3 warm pack')
    torch.manual_seed(seed)
    code_fingerprint = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                        (Path(__file__),Path(__file__).with_name('retarget_metrics.py'),Path(__file__).with_name('motion_lib.py'))}
    original = torch.load(source, map_location='cpu', weights_only=False)
    if original['dps'].shape[-1] != 69 or original['gts'].shape[-2] != 24:
        raise ValueError('Expected SMPL 69-coordinate / 24-body packed data')
    asset = PACKAGE_ROOT/'human_model_v2/assets/human_model_v2.xml'
    source_asset = PACKAGE_ROOT/'human_model_v1/assets/smpl_humanoid.xml'
    ki = extract_kinematic_info(str(asset)).to(torch.device(device))
    ski = extract_kinematic_info(str(source_asset))
    source_names = ski.body_names
    names_metadata = original.get('body_names')
    if names_metadata is not None and list(names_metadata) != source_names:
        raise ValueError('Source body ordering does not match SMPL contract')
    position_ids = [ki.body_names.index(n.replace('Ankle', 'Talus')) for n in source_names]
    rotation_ids = [ki.body_names.index(n) for n in source_names]
    feet = [ki.body_names.index(n) for n in FOOT_NAMES]
    source_feet = [source_names.index(n) for n in FOOT_NAMES]
    corners = box_corners(asset, FOOT_NAMES, device)
    source_corners = box_corners(source_asset, FOOT_NAMES, device)
    weights = torch.tensor([8. if any(k in n for k in ('Ankle','Toe','Hand')) else 1.
                            for n in source_names], device=device)[None, :, None]
    selected = list(range(len(original['motion_files']))) if motion_ids is None else list(motion_ids)
    if len(set(selected)) != len(selected) or any(i < 0 or i >= len(original['motion_files']) for i in selected):
        raise ValueError('Invalid or duplicate motion IDs')
    warm = torch.load(initial_pack,map_location='cpu',weights_only=False) if initial_pack else None
    if warm is not None:
        wm=warm.get('human_model_metadata',{})
        if wm.get('source_sha256') != hashlib.sha256(source.read_bytes()).hexdigest() or wm.get('asset_sha256') != hashlib.sha256(asset.read_bytes()).hexdigest():
            raise ValueError('Warm start source or model mismatch')
        warm_ids={r['motion_id']:i for i,r in enumerate(wm['motions'])}
        if wm.get('dof_names') != ki.dof_names or wm.get('body_names') != ki.body_names:
            raise ValueError('Warm start body/joint ordering mismatch')
        if iterations==0 and any(i not in warm_ids for i in selected):
            raise ValueError('Projection-only conversion requires every selected motion in warm pack')
    trust = torch.load(trust_reference_pack,map_location='cpu',weights_only=False) if trust_reference_pack else None
    if trust is not None:
        tm=trust.get('human_model_metadata',{})
        if (tm.get('source_sha256') != hashlib.sha256(source.read_bytes()).hexdigest()
                or tm.get('asset_sha256') != hashlib.sha256(asset.read_bytes()).hexdigest()
                or tm.get('dof_names') != ki.dof_names or tm.get('body_names') != ki.body_names):
            raise ValueError('Trust reference provenance mismatch')
        trust_ids={r['motion_id']:i for i,r in enumerate(tm['motions'])}
    fields = {k: [] for k in ('gts','grs','gvs','gavs','dps','dvs','contacts')}
    reports, frames, dts, files = [], [], [], []
    for mid in selected:
        start = int(original['length_starts'][mid]); n = int(original['motion_num_frames'][mid])
        dt = float(original['motion_dt'][mid]); sl = slice(start, start+n)
        if n < 3 or dt <= 0:
            raise ValueError(f'Invalid motion length/dt: {mid}')
        target = original['gts'][sl].to(device)
        source_quat = original['grs'][sl].to(device)
        if not torch.isfinite(target).all() or not torch.isfinite(source_quat).all():
            raise ValueError(f'Non-finite source: {mid}')
        if not torch.allclose(source_quat.norm(dim=-1), torch.ones_like(source_quat[...,0]), atol=1e-3):
            raise ValueError(f'Source rotations are not unit quaternions: {mid}')
        target, body_scale = normalize_skeleton(target, ski)
        target_rot = quaternion_to_matrix(source_quat, w_last=True)
        raw_contact = original['contacts'][sl, source_feet].to(device).bool()
        source_world = world_points(target, target_rot, source_feet, source_corners)
        source_bottom = source_world[...,2].min(-1).values
        # Source-only floor calibration BEFORE optimization. The raw mocap
        # pelvis trajectory and corrected target are distinct quantities.
        support_floor = torch.where(raw_contact, source_bottom, float('inf')).min(-1).values
        valid_floor=raw_contact.any(-1)
        if not valid_floor.any():
            raise ValueError(f'No source contact support for floor calibration: {mid}')
        # Interpolate the estimated floor during flight; do not pull airborne
        # feet to the ground just because both contacts are absent.
        import numpy as np
        valid_ids=torch.where(valid_floor)[0].cpu().numpy()
        support_floor=torch.as_tensor(np.interp(np.arange(n),valid_ids,support_floor[valid_floor].cpu().numpy()),device=device,dtype=target.dtype)
        import torch.nn.functional as F
        floor_correction = F.avg_pool1d(F.pad(support_floor[None,None],(2,2),mode='replicate'),5,stride=1)[0,0]
        target = target.clone()
        target[:,:,2] -= floor_correction[:,None]
        source_bottom = source_bottom-floor_correction[:,None]
        # Fixed source-only preprocessing, never output-driven relabeling.
        contact = raw_contact & (source_bottom <= .03)
        # Fixed skeleton in this packaged source: do not pretend individual
        # subject anthropometry is known. Segment-length mismatch is recorded.
        source_lengths = torch.stack([(target[:,i]-target[:,int(ski.parent_indices[i])]).norm(dim=-1).median()
                                      for i in range(1,24)])
        nominal = ski.local_pos[1:].norm(dim=-1).to(device)
        scale_ratio = source_lengths/nominal.clamp_min(1e-6)
        scale = float(scale_ratio.median())
        if abs(scale-1) > .05:
            raise ValueError(f'Uncalibrated source skeleton scale {scale:.3f}; provide compatible SMPL pack')
        local, anchors, phases = contact_targets(target, target_rot, source_feet, source_corners, corners, contact)
        q = torch.zeros(n, 59, device=device, requires_grad=True)
        root_delta = torch.zeros(n, 3, device=device, requires_grad=True)
        if warm is not None and mid in warm_ids:
            wi=warm_ids[mid];wa=int(warm['length_starts'][wi]);wn=int(warm['motion_num_frames'][wi])
            if wn != n:raise ValueError('Warm start frame count mismatch')
            with torch.no_grad():
                q.copy_(warm['dps'][wa:wa+n].to(device))
                root_delta.copy_(warm['gts'][wa:wa+n,0].to(device)-target[:,0])
                root_delta[:,:2]=0.
        source_steps=gait_metrics(target,target_rot,contact,[source_feet[0],source_feet[2]],dt)['steps']
        gait_feet = [ki.body_names.index('L_Talus'),ki.body_names.index('R_Talus')]
        if source_steps:
            step_a=torch.tensor([x['start'] for x in source_steps],device=device)
            step_b=torch.tensor([x['end'] for x in source_steps],device=device)
            step_side=torch.tensor([x['side'] for x in source_steps],device=device)
            step_target=torch.tensor([x['step_length_m'] for x in source_steps],device=device)
            step_tangent=derivative(target[:,0],dt)[(step_a+step_b)//2,:2]
            step_tangent=step_tangent/step_tangent.norm(dim=-1,keepdim=True).clamp_min(1e-6)
            step_feet=torch.tensor(gait_feet,device=device)
        opt = torch.optim.Adam([{'params':[q], 'lr':.04}, {'params':[root_delta], 'lr':.004}])
        planted = contact[1:] & contact[:-1]
        # Consecutive samples of a body have the same local proxy within a phase.
        for step in range(iterations):
            opt.zero_grad()
            pos, rot, _ = forward(ki, q, target[:,0]+root_delta, target_rot[:,0])
            xyz = world_points(pos, rot, feet, corners)
            points = material_points(pos, rot, feet, local)
            strength = min(1., step/100.)
            loss = (((pos[:,position_ids]-target)*weights).square()).mean()
            if source_steps:
                step_delta=pos[step_b,step_feet[step_side],:2]-pos[step_a,step_feet[1-step_side],:2]
                loss += 4.*((step_delta*step_tangent).sum(-1)-step_target).square().mean()
            loss += .015*(rot[:,rotation_ids]-target_rot).square().mean()
            loss += .002*(q[1:]-q[:-1]).square().mean()
            loss += .02*(q[2:]-2*q[1:-1]+q[:-2]).square().mean()
            loss += .02*((q[1:]-q[:-1]).abs()/dt-10.).clamp_min(0).square().mean()
            loss += 2.*root_delta.square().mean() + .02*(derivative(root_delta, dt)).square().mean()
            penetration_cost=(-xyz[...,2]+.001).clamp_min(0)
            loss += strength*400.*penetration_cost.square().mean()
            loss += strength*50.*penetration_cost.amax(dim=(1,2)).square().mean()
            loss += strength*10.*((points-anchors).square().sum(-1)*contact).sum()/contact.sum().clamp_min(1)
            drift = (points[1:,:,:2]-points[:-1,:,:2])/dt
            loss += strength*.08*(drift.square().sum(-1)*planted).sum()/planted.sum().clamp_min(1)
            loss.backward(); opt.step()
            with torch.no_grad():
                q.clamp_(ki.dof_limits_lower, ki.dof_limits_upper)
                root_delta.clamp_(-.08, .08)
                root_delta[:,:2]=0.  # preserve source horizontal turning trajectory exactly
            if step == int(iterations*.75):
                for group in opt.param_groups: group['lr'] *= .25
        if upper_refine_iterations:
            if upper_smoothing_hz is None or warm is None:
                raise ValueError('Upper refinement requires a warm pack and smoothing cutoff')
            refined=refine_upper_trajectory(ki,q,target[:,0]+root_delta.detach(),target,target_rot,
                                           position_ids,rotation_ids,dt,upper_smoothing_hz,upper_refine_iterations)
            with torch.no_grad():q.copy_(refined)
        with torch.no_grad():
            if upper_smoothing_hz is not None and not upper_refine_iterations:
                from .retarget_metrics import smooth_upper_hinges
                q.copy_(smooth_upper_hinges(q, ki.dof_names, dt, upper_smoothing_hz))
            root = target[:,0]+root_delta
            retained_correction=1.
            if trust is not None:
                ti=trust_ids[mid];ta=int(trust['length_starts'][ti])
                if int(trust['motion_num_frames'][ti])!=n or abs(float(trust['motion_dt'][ti])-dt)>1e-7:
                    raise ValueError('Trust reference timing mismatch')
                if not torch.allclose(root,trust['gts'][ta:ta+n,0].to(device),atol=1e-6,rtol=0):
                    raise ValueError('Trust reference root trajectory mismatch')
                projected,retained_correction=project_upper_correction(ki,q,trust['dps'][ta:ta+n].to(device),root,target_rot[:,0],upper_trust_radius_m)
                q.copy_(projected)
            trunk_report=None
            if source_trunk:
                from .retarget_metrics import source_trunk_trajectory
                if proximal_arms:
                    if not wm.get('source_trunk'):raise ValueError('Proximal fitting requires validated source-trunk warm data')
                    trunk_report=wm['motions'][warm_ids[mid]]['source_trunk']
                else:
                    locked,trunk_report=source_trunk_trajectory(q,target_rot,ski,ki,dt)
                    q.copy_(locked)
                if source_trunk_arm_iterations:
                    with torch.enable_grad():
                        arms=refine_upper_trajectory(ki,q,root.detach(),target,target_rot,position_ids,rotation_ids,dt,3.,source_trunk_arm_iterations,arms_only=True,source_ki=ski if proximal_arms else None)
                    q.copy_(arms)
            # Minimal upward translation projects sampled collision geometry
            # onto the floor inequality; q, root XY and heading stay unchanged.
            from protomotions.utils.rotations import slerp
            half_rot=quaternion_to_matrix(slerp(source_quat[:-1,0],source_quat[1:,0],torch.full((n-1,1),.5,device=device)),w_last=True)
            floor_lift=torch.zeros(n,device=device)
            for _ in range(2):
                fp,fr,_=forward(ki,q,root,target_rot[:,0])
                required=(.001-world_points(fp,fr,feet,corners)[...,2].amin(dim=(1,2))).clamp_min(0)
                hp,hr,_=forward(ki,(q[:-1]+q[1:])/2,(root[:-1]+root[1:])/2,half_rot)
                half_required=(.001-world_points(hp,hr,feet,corners)[...,2].amin(dim=(1,2))).clamp_min(0)
                required[:-1]=torch.maximum(required[:-1],half_required)
                required[1:]=torch.maximum(required[1:],half_required)
                root[:,2]+=required;floor_lift+=required
            pos, rot, _ = forward(ki, q, root, target_rot[:,0])
            vel = derivative(pos, dt)
            angvel = compute_angular_velocity(rot, 1/dt, 1)
            qd = derivative(q, dt)
            contacts = torch.zeros(n,26,device=device,dtype=torch.bool)
            for a, name in enumerate(source_names): contacts[:,ki.body_names.index(name)] = original['contacts'][sl,a].to(device).bool()
            contacts[:,feet] = contact
            err = (pos[:,position_ids]-target).norm(dim=-1)
            xyz = world_points(pos,rot,feet,corners)
            points = material_points(pos,rot,feet,local)
            cm = contact_metrics(points,phases,dt)
            # Fractional FK, not interpolation of independent body positions.
            from protomotions.utils.rotations import slerp
            root_quat = source_quat[:,0]
            half_rot = quaternion_to_matrix(slerp(root_quat[:-1],root_quat[1:],torch.full((n-1,1),.5,device=device)),w_last=True)
            hp, hr, _ = forward(ki,(q[:-1]+q[1:])/2,(root[:-1]+root[1:])/2,half_rot)
            half_bottom = world_points(hp,hr,feet,corners)[...,2].min()
            penetration = float((-torch.minimum(xyz[...,2].min(),half_bottom)).clamp_min(0))
            source_gait = gait_metrics(target,target_rot,contact,[source_feet[0],source_feet[2]],dt)
            gait_feet = [ki.body_names.index('L_Talus'), ki.body_names.index('R_Talus')]
            output_gait = gait_metrics(pos,rot,contact,gait_feet,dt,event_velocity=derivative(target[:,0],dt))
            step_errors=[]
            for step in source_gait['steps']:
                a,b,side=step['start'],step['end'],step['side']
                tangent=derivative(target[:,0],dt)[(a+b)//2,:2];tangent=tangent/tangent.norm().clamp_min(1e-6)
                got=float((pos[b,gait_feet[side],:2]-pos[a,gait_feet[1-side],:2])@tangent)
                step_errors.append(dict(error_m=abs(got-step['step_length_m']),limit_m=max(.02,.1*abs(step['step_length_m']))))
            root_error=(root[:,:2]-target[:,0,:2]).norm(dim=-1)
            report=dict(motion_id=mid,file=original['motion_files'][mid],frames=n,dt=dt,
                position_rmse_m=float(err.square().mean().sqrt()),foot_position_max_m=float(err[:,source_feet].max()),
                ground_penetration_max_m=penetration,contact_proxy=cm,
                max_joint_speed_rad_s=float(qd.abs().max()),joint_acceleration_p95_rad_s2=float(torch.quantile(derivative(qd,dt).abs().flatten(),.95)),
                root_path_rmse_m=float(root_error.square().mean().sqrt()),root_path_max_m=float(root_error.max()),
                root_height_change_max_m=float((root[:,2]-target[:,0,2]).abs().max()),floor_projection_max_m=float(floor_lift.max()),
                body_scale=body_scale,source_scale_median=scale,source_scale_spread_max=float((scale_ratio-scale).abs().max()),
                source_gait=source_gait,output_gait=output_gait,
                heading_error_max_deg=0.,turn_angle_error_deg=0.,yaw_rate_error_deg_s=0.,
                speed_mae_mps=float((vel[:,0,:2].norm(dim=-1)-derivative(target[:,0],dt)[:,:2].norm(dim=-1)).abs().mean()),
                source_floor_correction_max_m=float(floor_correction.abs().max()),
                source_support_fraction=float(contact.any(-1).float().mean()),
                source_contact_rejected_fraction=float((raw_contact & ~contact).sum()/raw_contact.sum().clamp_min(1)),
                finite=bool(torch.isfinite(qd).all() and torch.isfinite(pos).all()))
            from .retarget_metrics import temporal_jerk_metrics
            report['source_trunk'] = trunk_report
            report['proximal_arms'] = proximal_arms
            report['upper_correction_fraction'] = retained_correction
            report['temporal_quality'] = temporal_jerk_metrics(q, pos, rot, ki.dof_names, ki.body_names, dt)
            if 'source_anthropometry' in original:
                report['source_anthropometry'] = original['source_anthropometry'][mid]
            checks=dict(finite=report['finite'],rom=bool(((q>=ki.dof_limits_lower-1e-5)&(q<=ki.dof_limits_upper+1e-5)).all()),
                position=report['position_rmse_m']<=.04,foot_position=report['foot_position_max_m']<=.05,
                penetration=penetration<=.005,velocity=report['max_joint_speed_rad_s']<=15,
                slip=all(r['slip_p95_mps'] is not None and r['slip_p95_mps']<=.1 for r in cm),
                drift=all(r['drift_max_m'] is not None and r['drift_max_m']<=.02 for r in cm),
                root_path=report['root_path_rmse_m']<=.05 and report['root_path_max_m']<=.1,
                turning_step=bool(step_errors) and all(x['error_m']<=x['limit_m'] for x in step_errors),
                speed=report['speed_mae_mps']<=.1,
                cadence=source_gait['cadence_median_spm'] is not None and output_gait['cadence_median_spm'] is not None
                    and abs(output_gait['cadence_median_spm']-source_gait['cadence_median_spm'])<=.05*source_gait['cadence_median_spm'])
            if source_trunk:
                checks.pop('position')  # User prioritizes trunk/source rotation over hand position.
                checks['source_trunk']=all(x['source_envelope_pass'] and x['output_speed_max_rad_s']<=x['speed_limit_rad_s']*1.001+1e-5 and x['output_angular_jerk_max_rad_s3']<=x['jerk_limit_rad_s3']*1.001+1e-3 for x in trunk_report)
            report['checks']=checks
            report['failed_gates']=[k for k,v in checks.items() if not v]
            report['status']='screening_passed' if all(checks.values()) else 'candidate_failed_gate'
            reports.append(report)
            print(json.dumps({k:v for k,v in report.items() if k not in ('source_gait','output_gait')}),flush=True)
            for key,value in dict(gts=pos,grs=matrix_to_quaternion(rot,w_last=True),gvs=vel,gavs=angvel,dps=q,dvs=qd,contacts=contacts).items():
                fields[key].append(value.detach().cpu())
            frames.append(n); dts.append(dt); files.append(original['motion_files'][mid])
    if not frames: raise ValueError('No motions selected')
    pack={k:torch.cat(v) for k,v in fields.items()}
    counts=torch.tensor(frames,dtype=torch.long)
    pack.update(length_starts=torch.cat((torch.zeros(1,dtype=torch.long),counts.cumsum(0)[:-1])),
        motion_num_frames=counts,motion_dt=torch.tensor(dts),motion_lengths=(counts-1)*torch.tensor(dts),
        motion_weights=original['motion_weights'][selected].clone(),motion_files=tuple(files))
    pack['motion_weights']/=pack['motion_weights'].sum()
    meta=dict(converter='smpl2hm',model_id='human_model_v2',status='candidate_requires_confirmation',
        source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        asset_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),source_asset_sha256=hashlib.sha256(source_asset.read_bytes()).hexdigest(),
        dof_names=ki.dof_names,body_names=ki.body_names,iterations=iterations,seed=seed,
        contact_scope='source labels intersect source box bottom <=3cm; source-fixed corner proxy; no measured CoP',
        contact_height_threshold_m=.03,source_floor_calibration='5-frame moving average of source raw-contact lowest box height; translate whole skeleton vertically',
        anthropometry='segment-length normalization to fixed SMPL proxy, root displacement scaled by leg length ratio; subject mass/inertia unknown',
        joint_interpolation='scalar hinges with v2 FK reconstruction; requires HumanModelMotionLib',
        code_sha256=code_fingerprint,preserve_root_xy=True,
        initial_pack_sha256=hashlib.sha256(Path(initial_pack).read_bytes()).hexdigest() if initial_pack else None,
        criteria=dict(position_rmse_m=.04,foot_position_max_m=.05,penetration_max_m=.005,slip_p95_mps=.1,
                      contact_drift_max_m=.02,joint_speed_max_rad_s=15,root_path_rmse_m=.05,root_path_max_m=.1,speed_mae_mps=.1),
        trust_reference_sha256=hashlib.sha256(Path(trust_reference_pack).read_bytes()).hexdigest() if trust_reference_pack else None,
        upper_trust_radius_m=upper_trust_radius_m if trust_reference_pack else None,
        initial_pack=str(initial_pack) if initial_pack else None,
        source_trunk=source_trunk,source_trunk_arm_iterations=source_trunk_arm_iterations,proximal_arms=proximal_arms,upper_position_scope='diagnostic_only; source trunk prioritized' if source_trunk else 'original strict gate',
        upper_smoothing_hz=upper_smoothing_hz,upper_refine_iterations=upper_refine_iterations,physical_tracking='not_evaluated',motions=reports)
    pack['human_model_metadata']=meta
    destination.parent.mkdir(parents=True,exist_ok=True)
    torch.save(pack,destination);destination.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
    return meta



def render_comparison(source, converted_pack, destination, *, motion_ids=(0,12,13), duration=6., video_label=None):
    """Render stored world poses, not a dynamics rollout. Owned by smpl2hm e02."""
    import os
    os.environ.setdefault('MUJOCO_GL','egl')
    import mujoco
    import numpy as np
    import imageio_ffmpeg
    import subprocess
    from PIL import Image, ImageDraw, ImageFont
    from .retarget_metrics import FOOT_NAMES,box_corners,world_points
    source,converted_pack,destination=map(Path,(source,converted_pack,destination))
    if destination.exists():raise FileExistsError(destination)
    raw=torch.load(source,map_location='cpu',weights_only=False)
    target=torch.load(converted_pack,map_location='cpu',weights_only=False)
    meta=target['human_model_metadata']
    if hashlib.sha256(source.read_bytes()).hexdigest()!=meta['source_sha256']:
        raise ValueError('Comparison source differs from conversion source')
    kin=[extract_kinematic_info(str(PACKAGE_ROOT/x)) for x in
         ('human_model_v1/assets/smpl_humanoid.xml','human_model_v2/assets/human_model_v2.xml')]
    indices={r['motion_id']:i for i,r in enumerate(meta['motions'])}
    fps=30;width,height=1600,900
    fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=ImageFont.truetype(fontpath,22);title=ImageFont.truetype(fontpath,29)
    xml='<mujoco><visual><global offwidth="800" offheight="650"/><headlight ambient="0.6 0.6 0.6" diffuse="0.6 0.6 0.6"/></visual><worldbody><light pos="0 -3 5"/><geom type="plane" size="200 200 .1" rgba="0.16 0.19 0.24 1"/></worldbody></mujoco>'
    model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model)
    renderer=mujoco.Renderer(model,height=650,width=800)
    camera=mujoco.MjvCamera();camera.distance=3.8;camera.azimuth=135;camera.elevation=-16
    identity=np.eye(3).ravel();zero=np.zeros(3)
    def geom(kind,size,pos,color):
        scene=renderer.scene;g=scene.geoms[scene.ngeom];scene.ngeom+=1
        mujoco.mjv_initGeom(g,kind,np.asarray(size,dtype=float),np.asarray(pos,dtype=float),identity,np.asarray(color,dtype=np.float32))
        return g
    def line(a,b,color,width=.019):
        g=geom(mujoco.mjtGeom.mjGEOM_CAPSULE,zero,zero,color)
        mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_CAPSULE,width,np.asarray(a,dtype=float),np.asarray(b,dtype=float))
    destination.parent.mkdir(parents=True,exist_ok=True)
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','rawvideo','-pix_fmt','rgb24','-s','1600x900','-r',str(fps),'-i','-','-an','-c:v','libx264','-crf','19','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',str(destination)]
    count=0;segments=[]
    with destination.with_suffix('.render.log').open('w') as log:
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=log)
        try:
            for mid in motion_ids:
                ti=indices[mid];starts=[int(raw['length_starts'][mid]),int(target['length_starts'][ti])]
                source_dt=float(raw['motion_dt'][mid]);target_dt=float(target['motion_dt'][ti])
                if abs(source_dt-target_dt)>1e-6:
                    raise ValueError('Comparison requires matching source/target timestamps')
                available=min(int(raw['motion_num_frames'][mid]),int(target['motion_num_frames'][ti]))
                n=min(round(duration*fps),int((available-1)*source_dt*fps)+1)
                sample_indices=torch.round(torch.arange(n)/(fps*source_dt)).long().clamp_max(available-1)
                arrays=[raw,target];poses=[];feet=[]
                for j,pack in enumerate(arrays):
                    a=starts[j];pos=pack['gts'][a+sample_indices].clone()
                    # Same source initial XY origin; preserve original world Z.
                    pos[:,:,:2]-=raw['gts'][starts[0],0,:2]
                    rot=quaternion_to_matrix(pack['grs'][a+sample_indices],w_last=True)
                    asset=PACKAGE_ROOT/('human_model_v1/assets/smpl_humanoid.xml' if j==0 else 'human_model_v2/assets/human_model_v2.xml')
                    corners=box_corners(asset,FOOT_NAMES)
                    ids=[kin[j].body_names.index(x) for x in FOOT_NAMES]
                    feet.append(world_points(pos,rot,ids,corners).numpy());poses.append(pos.numpy())
                label=Path(raw['motion_files'][mid]).stem
                segments.append(dict(source_motion_id=mid,start_video_frame=count,frames=n,source_start_frame=0,label=label,source_dt=source_dt,source_frame_indices=sample_indices.tolist()))
                for frame in range(n):
                    canvas=Image.new('RGB',(width,height),(15,20,29));draw=ImageDraw.Draw(canvas)
                    draw.text((32,17),'SMPL -> human_model_v2 | synchronized motion comparison' + (f' | {video_label}' if video_label else ''),font=title,fill='white')
                    draw.text((32,64),f'{label}   |   source ID {mid}   |   t = {frame/fps:.2f} s   |   1x speed',font=font,fill='#cbd5e1')
                    for side in range(2):
                        xyz=poses[side][frame];camera.lookat[:]=[*poses[0][frame,0,:2],.85]
                        renderer.update_scene(data,camera=camera)
                        cx,cy=poses[0][frame,0,:2]
                        for g in range(-5,6):
                            line([np.floor(cx)+g,np.floor(cy)-5,.002],[np.floor(cx)+g,np.floor(cy)+5,.002],[.28,.33,.39,1],.002)
                            line([np.floor(cx)-5,np.floor(cy)+g,.002],[np.floor(cx)+5,np.floor(cy)+g,.002],[.28,.33,.39,1],.002)
                        for body,name in enumerate(kin[side].body_names):
                            color=[.2,.76,1,1] if name.startswith('L_') else ([1,.56,.27,1] if name.startswith('R_') else [.85,.9,.96,1])
                            if body and np.linalg.norm(xyz[body]-xyz[int(kin[side].parent_indices[body])])>1e-6:
                                line(xyz[int(kin[side].parent_indices[body])],xyz[body],color,.022)
                            geom(mujoco.mjtGeom.mjGEOM_SPHERE,[.035 if name!='Head' else .095,0,0],xyz[body],color)
                        for box in feet[side][frame]:
                            for a in range(8):
                                for b in range(a+1,8):
                                    if (a^b) in (1,2,4):line(box[a],box[b],[.68,.88,.62,1],.006)
                        canvas.paste(Image.fromarray(renderer.render()),(side*800,140))
                        draw.text((32+side*800,108),'SOURCE SMPL | 24 bodies' if side==0 else 'RETARGETED v2 | 26 bodies / 59 DOF',font=font,fill='white')
                    draw.line((800,105,800,790),fill='#596578',width=2)
                    draw.text((32,802),'Blue: left limbs   Orange: right limbs   Green: foot collision boxes',font=font,fill='#cbd5e1')
                    draw.text((32,840),'KINEMATIC PLAYBACK - not physics tracking. Raw source height retained; v2 includes floor correction.',font=font,fill='#f3cc84')
                    process.stdin.write(np.asarray(canvas).tobytes());count+=1
                    if frame==0:canvas.save(destination.parent.parent/'scratch'/f'comparison_source{mid}_preview.png')
                print('RENDERED',mid,n,flush=True)
        finally:
            process.stdin.close();renderer.close();returncode=process.wait()
        if returncode:raise RuntimeError(f'ffmpeg failed: {returncode}')
    report=dict(video_label=video_label,source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),converted_pack=str(converted_pack),converted_sha256=hashlib.sha256(converted_pack.read_bytes()).hexdigest(),fps=fps,frames=count,segments=segments,scope='Stored world-pose kinematic playback; original raw source height; identical view and source XY origin; schematic bones and actual foot box corners, not muscles/skin or physics tracking')
    destination.with_suffix('.comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(prog='smpl2hm')
    parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--iterations',type=int,default=800);parser.add_argument('--device',default='cpu')
    parser.add_argument('--motion-ids',type=int,nargs='+');parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--initial-pack')
    parser.add_argument('--upper-smoothing-hz',type=float)
    parser.add_argument('--upper-refine-iterations',type=int,default=0)
    parser.add_argument('--trust-reference-pack')
    parser.add_argument('--upper-trust-radius-m',type=float,default=.0295)
    parser.add_argument('--source-trunk',action='store_true')
    parser.add_argument('--proximal-arms',action='store_true',help='v4 candidate: frozen v3 trunk; source-angle/rate priority decreases distally')
    parser.add_argument('--source-trunk-arm-iterations',type=int,default=0)
    parser.add_argument('--render-comparison',action='store_true')
    parser.add_argument('--converted-pack')
    parser.add_argument('--duration',type=float,default=6.)
    parser.add_argument('--video-label',help='Display label only; does not change the retargeting implementation')
    args=parser.parse_args()
    if args.render_comparison:
        if not args.converted_pack:parser.error('--converted-pack is required for comparison')
        render_comparison(args.source,args.converted_pack,args.output,motion_ids=args.motion_ids or [0,12,13],duration=args.duration,video_label=args.video_label)
        return
    convert(args.source,args.output,iterations=args.iterations,device=args.device,motion_ids=args.motion_ids,seed=args.seed,initial_pack=args.initial_pack,upper_smoothing_hz=args.upper_smoothing_hz,upper_refine_iterations=args.upper_refine_iterations,trust_reference_pack=args.trust_reference_pack,upper_trust_radius_m=args.upper_trust_radius_m,source_trunk=args.source_trunk,source_trunk_arm_iterations=args.source_trunk_arm_iterations,proximal_arms=args.proximal_arms)


if __name__ == '__main__':
    main()
