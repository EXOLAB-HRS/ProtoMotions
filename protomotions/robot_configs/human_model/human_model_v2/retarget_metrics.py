"""smpl2hm geometry and gait measurements. Contact labels remain mocap proxies."""
import itertools
import xml.etree.ElementTree as ET
import numpy as np
import torch
from protomotions.utils.rotations import quaternion_to_matrix

FOOT_NAMES = ('L_Ankle', 'L_Toe', 'R_Ankle', 'R_Toe')


def box_corners(asset, names, device='cpu'):
    tree = ET.parse(asset)
    result = []
    for name in names:
        body = tree.find(f".//body[@name='{name}']")
        geoms = body.findall('geom')
        if len(geoms) != 1 or geoms[0].get('type') != 'box':
            raise ValueError(f'Expected one foot box: {name}')
        g = geoms[0]
        vector = lambda key, default: torch.tensor([float(x) for x in g.get(key, default).split()], device=device)
        corners = torch.tensor(list(itertools.product((-1., 1.), repeat=3)), device=device) * vector('size', '')
        rotation = quaternion_to_matrix(vector('quat', '1 0 0 0'), w_last=False)
        result.append(corners @ rotation.T + vector('pos', '0 0 0'))
    return torch.stack(result)


def world_points(pos, rot, ids, local):
    return pos[:, ids, None] + torch.einsum('tfij,fkj->tfki', rot[:, ids], local)


def intervals(mask):
    """Half-open contact runs; no clipping of inconvenient frames."""
    values = mask.detach().cpu().tolist()
    start = None
    for i, active in enumerate(values + [False]):
        if active and start is None:
            start = i
        elif not active and start is not None:
            yield start, i
            start = None


def contact_targets(source_pos, source_rot, source_ids, source_corners, target_corners, contact):
    """Source-fixed lowest-corner proxy per contact run, not measured CoP.

    Corner index is fixed before optimization. Split ankle/toe labels provide
    hindfoot/forefoot proxies, but do not establish physical contact validity.
    """
    world = world_points(source_pos, source_rot, source_ids, source_corners)
    n, feet = contact.shape
    local = torch.zeros(n, feet, 3, device=source_pos.device)
    anchor = torch.zeros_like(local)
    phases = []
    for foot in range(feet):
        for a, b in intervals(contact[:, foot]):
            # One fixed corner for a whole source phase. No output-driven relabeling.
            corner = int(world[a:b, foot, :, 2].mean(0).argmin())
            local[a:b, foot] = target_corners[foot, corner]
            anchor[a:b, foot, :2] = world[a:b, foot, corner, :2].median(0).values
            phases.append((foot, a, b, corner))
    return local, anchor, phases


def material_points(pos, rot, ids, local):
    return pos[:, ids] + torch.einsum('tfij,tfj->tfi', rot[:, ids], local)


def derivative(x, dt):
    if len(x) < 2 or dt <= 0:
        raise ValueError('Derivative requires >=2 samples and positive dt')
    return torch.cat(((x[1]-x[0])[None], (x[2:]-x[:-2])/2, (x[-1]-x[-2])[None]))/dt


def smooth_upper_hinges(q, names, dt, cutoff_hz):
    """Offline Gaussian low-pass on torso/arm hinges only; no root/leg edits.

    The cutoff is the Gaussian -3dB frequency in physical Hz. Odd reflection
    preserves endpoint values and linear trends. Projection into the original
    per-hinge range preserves ROM; all edge/clipping jerk remains audited.
    """
    import math
    import torch.nn.functional as F
    if dt <= 0 or not 0 < cutoff_hz < .5/dt:
        raise ValueError('Upper smoothing cutoff must be between zero and Nyquist')
    ids = [i for i, name in enumerate(names) if name.split('_')[0] in
           ('Torso', 'Spine', 'Chest', 'Neck', 'Head') or any(
               part in name for part in ('Thorax_', 'Shoulder_', 'Elbow_', 'Wrist_', 'Hand_'))]
    if not ids or q.ndim != 2 or q.shape[1] != len(names):
        raise ValueError('Upper smoothing requires named hinge trajectories')
    sigma = math.sqrt(math.log(2))/(2*math.pi*cutoff_hz*dt)
    radius = min(len(q)-1, max(1, math.ceil(4*sigma)))
    x = torch.arange(-radius, radius+1, dtype=q.dtype, device=q.device)
    kernel = torch.exp(-.5*(x/sigma).square()); kernel /= kernel.sum()
    values = q[:,ids].T[:,None]
    padded = F.pad(values, (radius,radius), mode='reflect')
    padded[:,:,:radius] = 2*values[:,:,:1]-padded[:,:,:radius]
    padded[:,:,-radius:] = 2*values[:,:,-1:]-padded[:,:,-radius:]
    filtered = F.conv1d(padded, kernel[None,None])[:,0].T
    filtered = filtered.maximum(q[:,ids].amin(0)).minimum(q[:,ids].amax(0))
    filtered[0] = q[0,ids]; filtered[-1] = q[-1,ids]
    result = q.clone(); result[:,ids] = filtered
    return result


def temporal_jerk_metrics(q, positions, rotations, dof_names, body_names, dt):
    """Full-clip forward differences, including edge stencils (no trimming).

    These are discrete sampled jerk measurements, not a claim that linear
    MotionLib interpolation is C3-continuous. Relative rotations remove pelvis
    motion; world-position jerk is also retained so root noise is not hidden.
    """
    from scipy.spatial.transform import Rotation
    if len(q) < 4 or dt <= 0:
        raise ValueError('Jerk needs at least four frames and positive dt')
    def stats(values):
        v=values.abs().double().flatten()
        return dict(p95=float(torch.quantile(v,.95)),max=float(v.max()))
    upper=[i for i,n in enumerate(dof_names) if n.split('_')[0] in
           ('Torso','Spine','Chest','Neck','Head') or any(k in n for k in ('Thorax_','Shoulder_','Elbow_','Wrist_','Hand_'))]
    body=[body_names.index(n) for n in ('Chest','L_Shoulder','R_Shoulder','L_Elbow','R_Elbow','L_Hand','R_Hand')]
    p=positions.double();r=rotations.double()
    relative=torch.einsum('tij,tbj->tbi',r[:,0].transpose(-1,-2),p[:,body]-p[:,0,None])
    rr=r[:,0,None].transpose(-1,-2)@r[:,body]
    delta=rr[1:]@rr[:-1].transpose(-1,-2)
    omega=torch.tensor(Rotation.from_matrix(delta.cpu().numpy().reshape(-1,3,3)).as_rotvec().reshape(len(q)-1,len(body),3))/dt
    angular_jerk=torch.diff(omega,n=2,dim=0)/dt**2
    return dict(upper_hinge_jerk_rad_s3=stats(torch.diff(q[:,upper].double(),n=3,dim=0)/dt**3),
                upper_relative_position_jerk_m_s3=stats(torch.diff(relative,n=3,dim=0).norm(dim=-1)/dt**3),
                upper_world_position_jerk_m_s3=stats(torch.diff(p[:,body],n=3,dim=0).norm(dim=-1)/dt**3),
                chest_shoulders_relative_angular_jerk_rad_s3=stats(angular_jerk[:,:3].norm(dim=-1)),
                root_position_jerk_m_s3=stats(torch.diff(p[:,0],n=3,dim=0).norm(dim=-1)/dt**3),
                scope='Forward finite differences; all valid stencils including endpoints; no physics/C3 continuity claim')


TRUNK_SEGMENTS = ('Torso','Spine','Chest','L_Thorax','R_Thorax')


def source_trunk_trajectory(q, source_rot, source_ki, target_ki, dt, cutoff_hz=2.):
    """Source-local trunk trajectory; hand-position IK cannot drive these hinges.

    Positive Gaussian averaging and clip-wide contraction keep each axis within
    the source envelope/plant ROM. Rotation speed and discrete angular jerk are
    limited by the source's own p95, measured separately for every segment.
    """
    import math
    from scipy.spatial.transform import Rotation
    result=q.detach().clone();rows=[]
    sigma=math.sqrt(math.log(2))/(2*math.pi*cutoff_hz*dt)
    radius=max(1,math.ceil(4*sigma));x=np.arange(-radius,radius+1)
    kernel=np.exp(-.5*(x/sigma)**2);kernel/=kernel.sum()
    sr=source_rot.detach().double().cpu().numpy()
    def kinematics(local):
        omega=Rotation.from_matrix(local[1:]@np.swapaxes(local[:-1],-1,-2)).as_rotvec()/dt
        jerk=np.linalg.norm(np.diff(omega,n=2,axis=0),axis=-1)/dt**2
        return np.linalg.norm(omega,axis=-1),jerk
    for name in TRUNK_SEGMENTS:
        si=source_ki.body_names.index(name);ti=target_ki.body_names.index(name)
        parent=int(source_ki.parent_indices[si]);tp=int(target_ki.parent_indices[ti])
        if source_ki.body_names[parent]!=target_ki.body_names[tp]:raise ValueError('Trunk topology differs')
        ref=target_ki.local_rot_ref_mat[ti].double().cpu().numpy()
        local=np.swapaxes(sr[:,parent],-1,-2)@sr[:,si]
        raw=np.unwrap(Rotation.from_matrix(ref.T@local).as_euler('XYZ'),axis=0)
        ids=[target_ki.dof_names.index(name+'_'+axis) for axis in 'xyz']
        if not torch.allclose(target_ki.hinge_axes_map[ti],torch.eye(3,device=q.device),atol=1e-6):raise ValueError('Trunk requires intrinsic XYZ hinges')
        lo=target_ki.dof_limits_lower[ids].cpu().numpy();hi=target_ki.dof_limits_upper[ids].cpu().numpy()
        # SMPL clavicle neutral angles exceed the fixed plant ROM. Preserve
        # the validated plant neutral and transfer source variation, explicitly.
        offset=(q[:,ids].detach().double().cpu().numpy().mean(0)-raw.mean(0)) if 'Thorax' in name else np.zeros(3)
        aligned=raw+offset
        bounded=np.clip(aligned,lo,hi)
        smooth=np.stack([np.convolve(np.pad(bounded[:,j],radius,mode='edge'),kernel,mode='valid') for j in range(3)],-1)
        center=smooth.mean(0);delta=smooth-center
        src_speed,src_jerk=kinematics(local)
        speed_limit=max(float(np.quantile(src_speed,.95)),1e-5)
        jerk_limit=max(float(np.quantile(src_jerk,.95)),1e-3)
        scale=1.
        for _ in range(12):
            candidate=center+scale*delta
            speed,jerk=kinematics(ref@Rotation.from_euler('XYZ',candidate).as_matrix())
            factor=min(1.,speed_limit/max(float(speed.max()),1e-12),jerk_limit/max(float(jerk.max()),1e-12))
            if factor>=.999999:break
            scale*=factor*.995
        result[:,ids]=torch.as_tensor(candidate,device=q.device,dtype=q.dtype)
        rows.append(dict(segment=name,source_axis_min_rad=raw.min(0).tolist(),source_axis_max_rad=raw.max(0).tolist(),neutral_offset_rad=offset.tolist(),aligned_source_min_rad=aligned.min(0).tolist(),aligned_source_max_rad=aligned.max(0).tolist(),
                         source_speed_p95_rad_s=float(np.quantile(src_speed,.95)),source_speed_max_rad_s=float(src_speed.max()),
                         source_angular_jerk_p95_rad_s3=float(np.quantile(src_jerk,.95)),source_angular_jerk_max_rad_s3=float(src_jerk.max()),
                         speed_limit_rad_s=speed_limit,jerk_limit_rad_s3=jerk_limit,variation_scale=scale,
                         output_speed_max_rad_s=float(speed.max()),output_angular_jerk_max_rad_s3=float(jerk.max()),
                         output_axis_min_rad=candidate.min(0).tolist(),output_axis_max_rad=candidate.max(0).tolist(),
                         source_envelope_pass=bool((candidate>=aligned.min(0)-1e-6).all() and (candidate<=aligned.max(0)+1e-6).all())))
    return result,rows



def source_arm_reference(source_rot, source_ki, target_ki, dt):
    """v4 candidate: common local XYZ frames, no shoulder neutral-offset fitting.

    A symmetric Gaussian preserves timing (no causal delay). ROM conflicts are
    clipped explicitly and must be assessed against RAW source in validation.
    Per-axis residual limits increase with distance from the trunk.
    """
    import math
    from scipy.spatial.transform import Rotation
    if dt <= 0:raise ValueError('Positive dt required')
    sr=source_rot.detach().double().cpu().numpy()
    q=torch.zeros(len(sr),target_ki.num_dofs,device=source_rot.device,dtype=source_rot.dtype)
    angle=torch.ones(target_ki.num_dofs,device=q.device,dtype=q.dtype)
    rate=angle.clone()
    sigma=math.sqrt(math.log(2))/(2*math.pi*4.*dt)
    radius=max(1,math.ceil(4*sigma));x=np.arange(-radius,radius+1)
    kernel=np.exp(-.5*(x/sigma)**2);kernel/=kernel.sum()
    rows=[]
    for part,angle_deg,rate_deg_s in [('Shoulder',2.,10.),('Elbow',5.,30.),('Wrist',10.,60.),('Hand',15.,90.)]:
        for side in ('L','R'):
            name=side+'_'+part;si=source_ki.body_names.index(name);ti=target_ki.body_names.index(name)
            sp=int(source_ki.parent_indices[si]);tp=int(target_ki.parent_indices[ti])
            if source_ki.body_names[sp]!=target_ki.body_names[tp]:raise ValueError('Arm topology differs')
            if not torch.allclose(target_ki.hinge_axes_map[ti].cpu(),torch.eye(3),atol=1e-6):raise ValueError('Intrinsic XYZ arm axes required')
            ids=[target_ki.dof_names.index(name+'_'+a) for a in 'xyz']
            local=np.swapaxes(sr[:,sp],-1,-2)@sr[:,si]
            raw=np.unwrap(Rotation.from_matrix(target_ki.local_rot_ref_mat[ti].double().cpu().numpy().T@local).as_euler('XYZ'),axis=0)
            lo=target_ki.dof_limits_lower[ids].cpu().numpy();hi=target_ki.dof_limits_upper[ids].cpu().numpy()
            bounded=np.clip(raw,lo,hi)
            smooth=np.stack([np.convolve(np.pad(bounded[:,j],radius,mode='edge'),kernel,mode='valid') for j in range(3)],-1)
            q[:,ids]=torch.as_tensor(smooth,device=q.device,dtype=q.dtype)
            angle[ids]=math.radians(angle_deg);rate[ids]=math.radians(rate_deg_s)
            rows.append(dict(segment=name,raw_rom_violation_max_deg=float(np.rad2deg(np.maximum(lo-raw,raw-hi).clip(0).max()))))
    return dict(q=q,angle_scale=angle,rate_scale=rate,rom_diagnostics=rows)

def contact_metrics(points, phases, dt):
    rows = []
    for side, foot_ids in (('left', (0, 1)), ('right', (2, 3))):
        speeds, drifts = [], []
        for foot, a, b, _ in phases:
            if foot not in foot_ids or b-a < 2:
                continue
            p = points[a:b, foot, :2]
            speeds.append(derivative(p, dt).norm(dim=-1))
            drifts.append((p-p[0]).norm(dim=-1).max())
        rows.append(dict(side=side, slip_p95_mps=float(torch.quantile(torch.cat(speeds), .95)) if speeds else None,
                         drift_max_m=float(torch.stack(drifts).max()) if drifts else None,
                         phase_count=len(drifts)))
    return rows


def gait_metrics(pos, rot, contact, foot_ids, dt, event_velocity=None):
    """Event-based (not frame-weighted) cadence and turning-local step length."""
    vel = derivative(pos[:, 0], dt)
    speed = vel[:, :2].norm(dim=-1)
    yaw = np.unwrap(torch.atan2(rot[:, 0, 1, 0], rot[:, 0, 0, 0]).detach().cpu().numpy())
    heading_rate = np.gradient(yaw, dt)
    if event_velocity is None:
        event_velocity=vel
    touches = []
    foot_contact = torch.stack((contact[:, :2].any(-1), contact[:, 2:].any(-1)), -1)
    # Source boundary contacts are not touchdown events.
    for side in (0, 1):
        last_end = 0
        for a, b in intervals(foot_contact[:, side]):
            if a > 0 and (a-last_end)*dt >= .1:
                touches.append((a, side))
            last_end = b
    touches.sort()
    cadence, lengths = [], [[], []]
    pairs = []
    for (a, left), (b, right) in zip(touches, touches[1:]):
        if left == right or not .15 <= (b-a)*dt <= 2:
            continue
        tangent = event_velocity[(a+b)//2, :2]
        if tangent.norm() < .2:
            continue
        tangent = tangent/tangent.norm()
        delta = pos[b, foot_ids[right], :2]-pos[a, foot_ids[left], :2]
        value = float(delta @ tangent)
        cadence.append(60/((b-a)*dt)); lengths[right].append(value)
        pairs.append(dict(start=a, end=b, side=right, step_length_m=value))
    return dict(speed_median_mps=float(speed.median()), yaw_change_deg=float(np.rad2deg(yaw[-1]-yaw[0])),
                yaw_rate_abs_p95_deg_s=float(np.quantile(np.abs(np.rad2deg(heading_rate)), .95)),
                cadence_median_spm=float(np.median(cadence)) if cadence else None,
                left_step_median_m=float(np.median(lengths[0])) if lengths[0] else None,
                right_step_median_m=float(np.median(lengths[1])) if lengths[1] else None,
                alternating_intervals=len(pairs), steps=pairs)


def normalize_skeleton(positions, kinematics):
    """Map observed segment lengths to fixed target SMPL proxy before fitting.

    Root horizontal displacement scales by mean leg length ratio; no temporal
    resampling. Not individualized anthropometry or mass/inertia calibration.
    """
    measured = torch.stack([(positions[:,i]-positions[:,int(kinematics.parent_indices[i])]).norm(dim=-1).median()
                            for i in range(1,kinematics.num_bodies)])
    nominal = kinematics.local_pos[1:].norm(dim=-1).to(positions.device)
    if not torch.isfinite(measured).all() or (measured < 1e-5).any():
        raise ValueError('Degenerate or non-finite source skeleton')
    leg_indices = [kinematics.body_names.index(n)-1 for n in ('L_Knee','L_Ankle','R_Knee','R_Ankle')]
    scale = nominal[leg_indices].sum()/measured[leg_indices].sum()
    if not .5 <= float(scale) <= 2:
        raise ValueError('Source units/body mapping not supported (leg scale outside .5..2)')
    root = positions[:,0].clone()*scale
    root[:,:2] += positions[0,0,:2]*(1-scale)
    result = [root]
    for b in range(1,kinematics.num_bodies):
        parent = int(kinematics.parent_indices[b])
        result.append(result[parent]+(positions[:,b]-positions[:,parent])*nominal[b-1]/measured[b-1])
    return torch.stack(result,1), dict(root_translation_scale=float(scale),
        segment_scales=(nominal/measured).tolist(),source_segment_lengths_m=measured.tolist())


def inventory(source, output):
    """Source-only condition inventory; not a retargeting pass certificate."""
    import json
    from pathlib import Path
    from protomotions.components.pose_lib import extract_kinematic_info
    from ..common.paths import PACKAGE_ROOT
    data=torch.load(source,map_location='cpu',weights_only=False)
    ki=extract_kinematic_info(str(PACKAGE_ROOT/'human_model_v1/assets/smpl_humanoid.xml'))
    ids=[ki.body_names.index(n) for n in FOOT_NAMES]
    rows=[]
    for mid,file in enumerate(data['motion_files']):
        a=int(data['length_starts'][mid]);n=int(data['motion_num_frames'][mid]);sl=slice(a,a+n)
        pos=data['gts'][sl];rot=quaternion_to_matrix(data['grs'][sl],w_last=True)
        m=gait_metrics(pos,rot,data['contacts'][sl][:,ids].bool(),[ids[0],ids[2]],float(data['motion_dt'][mid]))
        name=Path(file).name.lower()
        label=('turn' if 'turn' in name or 'around' in name or 'change' in name else
               'lateral' if 'side_step' in name else 'backward' if 'back' in name else
               'start_stop' if 'stand' in name else 'forward')
        rows.append(dict(motion_id=mid,file=file,source_class=label,**m))
    result=dict(source=str(source),clips=len(rows),duration_s=float(data['motion_lengths'].sum()),
                motions=rows,status='source_inventory_not_validation',subject_identity='folder labels are not guaranteed unique people')
    Path(output).write_text(json.dumps(result,indent=2)+'\n')
    return result


def audit_pack(pack_path, output_path, *, upper_position_diagnostic=False):
    """Independent saved-pack/FK/geometry audit including fractional timestamps."""
    import hashlib,json
    from pathlib import Path
    from protomotions.components.motion_lib import MotionLibConfig
    from protomotions.components.pose_lib import extract_kinematic_info, fk_batch_mjcf_with_velocities
    from ..common.paths import PACKAGE_ROOT
    from .motion_lib import HumanModelMotionLib
    data=torch.load(pack_path,map_location='cpu',weights_only=False)
    if upper_position_diagnostic and not data['human_model_metadata'].get('source_trunk'):
        raise ValueError('Upper position diagnostic mode is restricted to explicit source-trunk packs')
    source=torch.load(data['human_model_metadata']['source'],map_location='cpu',weights_only=False)
    source_asset=PACKAGE_ROOT/'human_model_v1/assets/smpl_humanoid.xml'
    ski=extract_kinematic_info(str(source_asset))
    source_feet=[ski.body_names.index(n) for n in FOOT_NAMES]
    source_corners=box_corners(source_asset,FOOT_NAMES)
    lib=HumanModelMotionLib(MotionLibConfig(motion_file=str(pack_path)))
    ki=lib.kinematics
    corners=box_corners(PACKAGE_ROOT/'human_model_v2/assets/human_model_v2.xml',FOOT_NAMES)
    feet=[ki.body_names.index(n) for n in FOOT_NAMES]
    rows=[]
    for mid,n in enumerate(data['motion_num_frames']):
        n=int(n);a=int(data['length_starts'][mid]);dt=float(data['motion_dt'][mid])
        q=data['dps'][a:a+n];root=data['gts'][a:a+n,0];quat=data['grs'][a:a+n,0]
        qpos=torch.cat((root,quat[:,[3,0,1,2]],q),-1)
        canonical=fk_batch_mjcf_with_velocities(ki,qpos,compute_velocities=False)
        stored_error=float((canonical.rigid_body_pos-data['gts'][a:a+n]).norm(dim=-1).max())
        r0=quaternion_to_matrix(canonical.rigid_body_rot,w_last=True)
        r1=quaternion_to_matrix(data['grs'][a:a+n],w_last=True)
        stored_rotation_error=float((r0-r1).abs().max())
        times=torch.arange(2*n-1)*dt/2
        state=lib.get_motion_state(torch.full((len(times),),mid,dtype=torch.long),times)
        qp=torch.cat((state.rigid_body_pos[:,0],state.rigid_body_rot[:,0,[3,0,1,2]],state.dof_pos),-1)
        independent=fk_batch_mjcf_with_velocities(ki,qp,compute_velocities=False)
        query_error=float((state.rigid_body_pos-independent.rigid_body_pos).norm(dim=-1).max())
        xyz=world_points(state.rigid_body_pos,quaternion_to_matrix(state.rigid_body_rot,w_last=True),feet,corners)
        penetration=float((-xyz[...,2]).clamp_min(0).max())
        finite=bool(all(torch.isfinite(x).all() for x in (state.dof_pos,state.dof_vel,state.rigid_body_pos,state.rigid_body_vel,state.rigid_body_ang_vel)))
        rom=float(torch.maximum(ki.dof_limits_lower-state.dof_pos,state.dof_pos-ki.dof_limits_upper).clamp_min(0).max())
        source_id=data['human_model_metadata']['motions'][mid]['motion_id']
        sa=int(source['length_starts'][source_id])
        target,_=normalize_skeleton(source['gts'][sa:sa+n],ski)
        source_rot=quaternion_to_matrix(source['grs'][sa:sa+n],w_last=True)
        raw_contact=source['contacts'][sa:sa+n][:,source_feet].bool()
        bottom=world_points(target,source_rot,source_feet,source_corners)[...,2].amin(-1)
        floor=torch.where(raw_contact,bottom,float('inf')).amin(-1)
        valid=raw_contact.any(-1);vid=torch.where(valid)[0].numpy()
        import torch.nn.functional as F
        floor=torch.tensor(np.interp(np.arange(n),vid,floor[valid].numpy()),dtype=target.dtype)
        floor=F.avg_pool1d(F.pad(floor[None,None],(2,2),mode='replicate'),5,stride=1)[0,0]
        target[:,:,2]-=floor[:,None]
        dense_target=torch.empty(len(times),24,3);dense_target[::2]=target;dense_target[1::2]=(target[:-1]+target[1:])/2
        mapped=[ki.body_names.index(name.replace('Ankle','Talus')) for name in ski.body_names]
        error=(state.rigid_body_pos[:,mapped]-dense_target).norm(dim=-1)
        body_rmse=float(error.square().mean().sqrt());foot_max=float(error[:,source_feet].max())
        # Heading and horizontal path are fixed in the latest converter, but
        # verify from the saved file rather than relying on that construction.
        path_error=float((state.rigid_body_pos[:,0,:2]-dense_target[:,0,:2]).norm(dim=-1).max())
        heading_error=float((quaternion_to_matrix(data['grs'][a:a+n,0],w_last=True)-source_rot[:,0]).abs().max())
        checks=dict(stored_fk=stored_error<=1e-5,stored_rotation=stored_rotation_error<=1e-4,
                    interpolated_fk=query_error<=1e-5,finite=finite,rom=rom<=1e-5,penetration=penetration<=.005,
                    position=body_rmse<=.04,foot_position=foot_max<=.05,query_velocity=float(state.dof_vel.abs().max())<=15,
                    root_path=path_error<=.1,root_heading_matrix=heading_error<=1e-4)
        if upper_position_diagnostic:checks.pop('position')
        rows.append(dict(motion_id=data['human_model_metadata']['motions'][mid]['motion_id'],
                         stored_fk_max_m=stored_error,stored_rotation_matrix_max=stored_rotation_error,
                         position_rmse_m=body_rmse,foot_position_max_m=foot_max,root_path_max_m=path_error,root_heading_matrix_max=heading_error,
                         fractional_fk_max_m=query_error,penetration_max_m=penetration,rom_excess_rad=rom,
                         max_query_joint_velocity_rad_s=float(state.dof_vel.abs().max()),
                         checks=checks,status='passed' if all(checks.values()) else 'candidate_failed_gate'))
    result=dict(upper_position_diagnostic=upper_position_diagnostic,source=str(pack_path),sha256=hashlib.sha256(Path(pack_path).read_bytes()).hexdigest(),motions=rows,
                scope='Canonical FK, geometry and corrected-target error at stored/half-frame times; contact/gait/physics not certified',
                status='passed' if all(r['status']=='passed' for r in rows) else 'candidate_failed_gate')
    Path(output_path).write_text(json.dumps(result,indent=2)+'\n')
    return result


def merge_candidates(paths, destination, *, include_ids=None):
    """Merge deterministic shards without promoting or hiding failed trials."""
    import copy,hashlib,json
    from pathlib import Path
    destination=Path(destination)
    if destination.exists() or destination.with_suffix('.json').exists():
        raise FileExistsError(destination)
    entries={};provenance=[];base=None
    for path in paths:
        data=torch.load(path,map_location='cpu',weights_only=False)
        meta=data['human_model_metadata']
        if base is None:base=copy.deepcopy(meta)
        for key in ('model_id','source_sha256','asset_sha256','dof_names','body_names','code_sha256'):
            if meta.get(key)!=base.get(key):raise ValueError(f'Incompatible shard {key}')
        provenance.append(dict(path=str(path),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()))
        for i,row in enumerate(meta['motions']):
            mid=row['motion_id']
            if mid in entries:raise ValueError(f'Duplicate source ID {mid}')
            a=int(data['length_starts'][i]);n=int(data['motion_num_frames'][i]);sl=slice(a,a+n)
            entries[mid]=(row,{k:data[k][sl] for k in ('gts','grs','gvs','gavs','dps','dvs','contacts')},
                          n,float(data['motion_dt'][i]),data['motion_files'][i])
    selected=sorted(entries) if include_ids is None else sorted(include_ids)
    if not selected or any(i not in entries for i in selected):raise ValueError('Invalid merge selection')
    pack={k:torch.cat([entries[i][1][k] for i in selected]) for k in entries[selected[0]][1]}
    counts=torch.tensor([entries[i][2] for i in selected]);dts=torch.tensor([entries[i][3] for i in selected])
    pack.update(motion_num_frames=counts,length_starts=torch.cat((torch.zeros(1,dtype=torch.long),counts.cumsum(0)[:-1])),
                motion_dt=dts,motion_lengths=(counts-1)*dts,motion_weights=torch.ones(len(selected))/len(selected),
                motion_files=tuple(entries[i][4] for i in selected))
    base.update(motions=[entries[i][0] for i in selected],shards=provenance,
                status='candidate_requires_confirmation',selection_scope='Explicit source IDs; uniform clip sampling, not balanced teacher distribution')
    pack['human_model_metadata']=base
    destination.parent.mkdir(parents=True,exist_ok=True)
    torch.save(pack,destination);destination.with_suffix('.json').write_text(json.dumps(base,indent=2)+'\n')
    return base


def condition_summary(pack_path, output_path):
    """Descriptive paired cadence/length per source speed and turn condition.

    Summary units are clip medians, never frames as independent subjects.
    No population SD or normative band is inferred from this small motion pack.
    """
    import json
    from pathlib import Path
    from protomotions.components.pose_lib import extract_kinematic_info
    from ..common.paths import PACKAGE_ROOT
    data=torch.load(pack_path,map_location='cpu',weights_only=False);meta=data['human_model_metadata']
    source=torch.load(meta['source'],map_location='cpu',weights_only=False)
    ski=extract_kinematic_info(str(PACKAGE_ROOT/'human_model_v1/assets/smpl_humanoid.xml'))
    groups={};events=[]
    centers=np.array([.5,.75,1.,1.25,1.5])
    for row in meta['motions']:
        mid=row['motion_id'];a=int(source['length_starts'][mid]);n=int(source['motion_num_frames'][mid]);dt=float(source['motion_dt'][mid])
        target,_=normalize_skeleton(source['gts'][a:a+n],ski)
        velocity=derivative(target[:,0],dt)[:,:2];speed=velocity.norm(dim=-1)
        heading=quaternion_to_matrix(source['grs'][a:a+n,0],w_last=True)[:,:2,0]
        alignment=(heading*velocity).sum(-1)/(heading.norm(dim=-1)*speed).clamp_min(1e-6)
        import torch.nn.functional as F
        smooth_speed=F.avg_pool1d(F.pad(speed[None,None],(2,2),mode='replicate'),5,stride=1)[0,0]
        acceleration=derivative(smooth_speed,dt).abs()
        path=np.unwrap(np.arctan2(velocity[:,1].numpy(),velocity[:,0].numpy()))
        turn=np.rad2deg(np.gradient(path,dt))
        for step in row['source_gait']['steps']:
            l,h=step['start'],step['end'];v=float(speed[l:h+1].median())
            if v<.2:continue
            nearest=int(abs(centers-v).argmin());speed_bin=str(centers[nearest]) if abs(centers[nearest]-v)<=.125 else 'outside_0.5_1.5'
            rate=float(np.median(turn[l:h+1]));condition='straight' if abs(rate)<=5 else ('left' if rate>0 else 'right')+('_gentle' if abs(rate)<=30 else '_sharp')
            align=float(alignment[l:h+1].median())
            direction='forward' if align>=.85 else ('backward' if align<=-.85 else 'lateral_or_mixed')
            accel=float(torch.quantile(acceleration[l:h+1],.95))
            phase='steady' if accel<=.5 else 'transition'
            e=dict(source_motion_id=mid,speed_bin=speed_bin,condition=condition,direction=direction,phase=phase,acceleration_abs_p95_mps2=accel,speed_mps=v,path_turn_deg_s=rate,
                   cadence_spm=60/((h-l)*dt),step_length_m=step['step_length_m'],side=step['side'],start_frame=l,end_frame=h)
            events.append(e);groups.setdefault((speed_bin,condition,direction,phase),{}).setdefault(mid,[]).append(e)
    rows=[]
    for (speed_bin,condition,direction,phase),by_clip in sorted(groups.items()):
        paired=np.array([[np.median([x['cadence_spm'] for x in es]),np.median([x['step_length_m'] for x in es])] for es in by_clip.values()])
        rows.append(dict(speed_bin=speed_bin,condition=condition,direction=direction,phase=phase,clip_count=len(by_clip),step_count=sum(map(len,by_clip.values())),
                         source_motion_ids=sorted(by_clip),clip_median_pairs=paired.tolist(),
                         mean_cadence_spm=float(paired[:,0].mean()),mean_step_length_m=float(paired[:,1].mean()),
                         between_clip_sd_cadence_spm=float(paired[:,0].std(ddof=1)) if len(paired)>1 else None,
                         between_clip_sd_step_length_m=float(paired[:,1].std(ddof=1)) if len(paired)>1 else None))
    result=dict(source=str(pack_path),conditions=rows,events=events,scope='Source contact-proxy events, descriptive between-clip statistics; not independent subjects or normative mean/SD; direction cosine +/-0.85 and steady 5-frame-smoothed |speed acceleration| p95<=0.5 m/s2 are engineering labels, not biological limits')
    Path(output_path).write_text(json.dumps(result,indent=2)+'\n')
    return result
