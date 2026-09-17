"""v2 scalar-coordinate interpolation with consistent body FK and velocities."""
import hashlib
from pathlib import Path
import torch
from protomotions.components.motion_lib import MotionLib
from protomotions.components.pose_lib import extract_kinematic_info
from protomotions.utils.rotations import quaternion_to_matrix, matrix_to_quaternion
from ..common.paths import PACKAGE_ROOT
from .retarget import forward


class HumanModelMotionLib(MotionLib):
    """Only fixed v2 packs; preserve the generic/SMPL MotionLib unchanged."""
    def __init__(self, config, device='cpu'):
        asset = PACKAGE_ROOT/'human_model_v2/assets/human_model_v2.xml'
        self.kinematics = extract_kinematic_info(str(asset)).to(torch.device(device))
        if config.motion_file is None:
            raise ValueError('v2 MotionLib requires a motion pack')
        metadata = torch.load(config.motion_file, map_location='cpu', weights_only=False).get('human_model_metadata', {})
        if (metadata.get('model_id') != 'human_model_v2'
                or metadata.get('asset_sha256') != hashlib.sha256(asset.read_bytes()).hexdigest()
                or metadata.get('body_names') != self.kinematics.body_names
                or metadata.get('dof_names') != self.kinematics.dof_names):
            raise ValueError('v2 motion provenance/order/asset mismatch')
        super().__init__(config, device)
        if self.lrs is not None or self.dps.shape[-1] != 59 or self.gts.shape[1] != 26:
            raise ValueError('v2 requires 59 scalar hinges, 26 bodies, no SMPL local rotations')

    def _pose(self, ids, times):
        state = super().get_motion_state(ids, times)
        pos, rot, _ = forward(self.kinematics, state.dof_pos,
                             state.rigid_body_pos[:,0],
                             quaternion_to_matrix(state.rigid_body_rot[:,0], w_last=True))
        state.rigid_body_pos = pos
        state.rigid_body_rot = matrix_to_quaternion(rot, w_last=True)
        return state, rot

    def get_motion_state(self, motion_ids, motion_times, joint_3d_format='exp_map'):
        time = motion_times.clamp_min(0).minimum(self.motion_lengths[motion_ids])
        state, _ = self._pose(motion_ids, time)
        h = self.motion_dt[motion_ids]*.03
        lo = (time-h).clamp_min(0)
        hi = (time+h).minimum(self.motion_lengths[motion_ids])
        span = (hi-lo).clamp_min(1e-7)
        before, r0 = self._pose(motion_ids, lo)
        after, r1 = self._pose(motion_ids, hi)
        state.rigid_body_vel = (after.rigid_body_pos-before.rigid_body_pos)/span[:,None,None]
        state.dof_vel = (after.dof_pos-before.dof_pos)/span[:,None]
        # World angular velocity from relative rotation's skew part; O(h^2).
        delta = r1 @ r0.transpose(-1,-2)
        skew = torch.stack((delta[...,2,1]-delta[...,1,2],
                            delta[...,0,2]-delta[...,2,0],
                            delta[...,1,0]-delta[...,0,1]),-1)
        state.rigid_body_ang_vel = skew/(2*span[:,None,None])
        # Event timing is defined by source frames; OR would advance touchdowns.
        idx0, _, _ = self._calc_frame_blend_from_id_and_time(motion_ids,time)
        contacts = self.get_motion_state_exact_frame(motion_ids,idx0).rigid_body_contacts
        state.rigid_body_contacts = contacts
        return state
