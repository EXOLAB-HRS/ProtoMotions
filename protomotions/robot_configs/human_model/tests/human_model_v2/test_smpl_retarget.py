"""SMPL to v2 motion conversion contracts; a 69-coordinate clip must not survive unchanged."""
import json

import pytest
import torch

from protomotions.components.motion_lib import MotionLib, MotionLibConfig
from protomotions.components.pose_lib import (
    compute_forward_kinematics_from_transforms,
    extract_kinematic_info,
    extract_transforms_from_qpos,
)
from protomotions.robot_configs.human_model.human_model_v2.adapters import smpl_retarget
from protomotions.utils.rotations import matrix_to_quaternion

FRAMES = 16
FPS = 30.0


@pytest.fixture(scope="module")
def source_info():
    return extract_kinematic_info(str(smpl_retarget.SOURCE_MJCF)).to(
        torch.device("cpu"), torch.float32
    )


@pytest.fixture(scope="module")
def target_info():
    return extract_kinematic_info(str(smpl_retarget.TARGET_MJCF)).to(
        torch.device("cpu"), torch.float32
    )


@pytest.fixture(scope="module")
def source_library(source_info, tmp_path_factory):
    torch.manual_seed(20260917)
    lower, upper = source_info.dof_limits_lower, source_info.dof_limits_upper
    coords = lower + (upper - lower) * torch.rand(FRAMES, source_info.nq - 7)
    coords *= torch.linspace(0.2, 0.8, FRAMES).unsqueeze(-1)
    root_pos = torch.zeros(FRAMES, 3)
    root_pos[:, 2] = 0.95
    root_pos[:, 0] = torch.linspace(0.0, 0.5, FRAMES)
    root_quat = torch.zeros(FRAMES, 4)
    root_quat[:, 0] = 1.0

    qpos = torch.cat([root_pos, root_quat, coords], dim=-1)
    root, joint_mats = extract_transforms_from_qpos(source_info, qpos)
    world_pos, world_rot = compute_forward_kinematics_from_transforms(source_info, root, joint_mats)

    bodies = len(source_info.body_names)
    contacts = torch.zeros(FRAMES, bodies, dtype=torch.bool)
    contacts[:, source_info.body_names.index("L_Toe")] = True

    payload = {
        "gts": world_pos,
        "grs": matrix_to_quaternion(world_rot, w_last=True),
        "gvs": torch.zeros(FRAMES, bodies, 3),
        "gavs": torch.zeros(FRAMES, bodies, 3),
        "dps": coords,
        "dvs": torch.zeros_like(coords),
        "lrs": matrix_to_quaternion(joint_mats, w_last=True),
        "contacts": contacts,
        "length_starts": torch.zeros(1, dtype=torch.long),
        "motion_lengths": torch.tensor([(FRAMES - 1) / FPS]),
        "motion_dt": torch.tensor([1.0 / FPS]),
        "motion_num_frames": torch.tensor([FRAMES]),
        "motion_weights": torch.ones(1),
        "motion_files": ("synthetic_smpl.motion",),
    }
    path = tmp_path_factory.mktemp("retarget") / "source.pt"
    torch.save(payload, path)
    return path


def test_differentiable_fk_matches_pose_lib(target_info):
    torch.manual_seed(7)
    lower, upper = target_info.dof_limits_lower, target_info.dof_limits_upper
    coords = lower + (upper - lower) * torch.rand(5, target_info.nq - 7)
    root_pos = torch.randn(5, 3)
    root_quat = torch.nn.functional.normalize(torch.randn(5, 4), dim=-1)
    qpos = torch.cat([root_pos, root_quat, coords], dim=-1)

    root, joint_mats = extract_transforms_from_qpos(target_info, qpos)
    expected_pos, expected_rot = compute_forward_kinematics_from_transforms(
        target_info, root, joint_mats
    )
    actual_pos, actual_rot = smpl_retarget._forward_kinematics(
        target_info, root_pos, joint_mats[:, 0], coords
    )
    torch.testing.assert_close(actual_pos, expected_pos)
    torch.testing.assert_close(actual_rot, expected_rot)

    tracked = coords.clone().requires_grad_(True)
    smpl_retarget._forward_kinematics(target_info, root_pos, joint_mats[:, 0], tracked)[0].sum().backward()
    assert tracked.grad is not None and bool(torch.isfinite(tracked.grad).all())
    assert float(tracked.grad.abs().sum()) > 0


def test_retarget_emits_v2_coordinates_without_local_rotations(source_library, target_info, tmp_path):
    output = tmp_path / "retargeted.pt"
    report_path = tmp_path / "report.json"
    report = smpl_retarget.retarget_motion_library(
        source_library,
        output,
        report_path,
        smpl_retarget.RetargetConfig(rotation_steps=25, leg_steps=10),
    )

    saved = torch.load(output, map_location="cpu", weights_only=False)
    assert "lrs" not in saved
    assert saved["dps"].shape == (FRAMES, target_info.nq - 7)
    assert saved["gts"].shape == (FRAMES, len(target_info.body_names), 3)
    assert saved["contacts"].shape == (FRAMES, len(target_info.body_names))
    talus = target_info.body_names.index("L_Talus")
    calcaneus = target_info.body_names.index("L_Ankle")
    assert bool((saved["contacts"][:, talus] == saved["contacts"][:, calcaneus]).all())

    within_rom = (saved["dps"] >= target_info.dof_limits_lower - 1e-6) & (
        saved["dps"] <= target_info.dof_limits_upper + 1e-6
    )
    assert bool(within_rom.all())
    assert report["summary"]["rom_violation_max_rad"] == 0.0
    assert json.loads(report_path.read_text())["target_coordinates"] == target_info.nq - 7


def test_retargeted_library_loads_through_motion_lib(source_library, target_info, tmp_path):
    output = tmp_path / "retargeted.pt"
    smpl_retarget.retarget_motion_library(
        source_library, output, None, smpl_retarget.RetargetConfig(rotation_steps=25, leg_steps=10)
    )

    library = MotionLib(MotionLibConfig(motion_file=str(output)), device="cpu")
    state = library.get_motion_state(torch.zeros(4, dtype=torch.long), torch.full((4,), 0.1))
    dof_pos, body_pos = state.dof_pos, state.rigid_body_pos
    assert dof_pos is not None and body_pos is not None
    assert dof_pos.shape == (4, target_info.nq - 7)
    assert body_pos.shape == (4, len(target_info.body_names), 3)


def test_source_library_without_local_rotations_is_rejected(source_library, tmp_path):
    payload = torch.load(source_library, map_location="cpu", weights_only=False)
    payload.pop("lrs")
    broken = tmp_path / "no_lrs.pt"
    torch.save(payload, broken)
    with pytest.raises(ValueError, match="local rotations"):
        smpl_retarget.retarget_motion_library(broken, tmp_path / "out.pt")
