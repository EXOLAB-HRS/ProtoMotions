"""Anchor-free ⑥: balance from any standing stance, with no target pose.

Sibling of ``mlp_posture_foundation.py``, deliberately kept as a separate
experiment so the working anchored policy stays reproducible.  Differences:

* every episode resets to a *different* real standing stance drawn from the mined
  pose bank, and the PD target is that same stance -- so zero action holds
  whatever the robot was handed, and ⑥ can only earn reward by correcting it
* the reward references no pose at all (height, uprightness, capture point)
* AMP is off: with resets spread across many stances there is no single clip to
  imitate, and a walking clip would turn ⑥ into a second locomotion planner
* the anti-chatter *termination* is kept, because losing the nominal-pose term
  removes the regularizer that was implicitly holding chatter down

Success criterion: ``scripts/hc_eval_pose_bank.py`` with ``HC_BANK_PD=self`` must
beat the anchored policy's 52/64.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import torch

from human_controller.posture import (
    compute_posture_fall_termination,
    compute_posture_vibration_termination,
)
from human_controller.posture_free import (
    BALANCE_OBS_DIM,
    FORCE_OBS_DIM,
    compute_capture_point_reward,
    compute_com_stability_reward,
    compute_free_posture_observation,
    compute_motion_cleanliness_reward,
    compute_pelvis_height_reward,
    compute_support_margin_reward,
)
from protomotions.envs.context_views import EnvContext
from protomotions.envs.control.steering_control import SteeringControlConfig
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.base_env.env import BaseEnv
from protomotions.simulator.base_simulator.config import (
    DomainRandomizationConfig,
    PushDomainRandomizationConfig,
)

_BASE_PATH = (
    Path(__file__).resolve().parents[2]
    / "ProtoMotions"
    / "examples"
    / "experiments"
    / "steering"
    / "mlp.py"
)
_SPEC = importlib.util.spec_from_file_location("posture_free_base", _BASE_PATH)
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

scene_lib_config = _BASE.scene_lib_config
motion_lib_config = _BASE.motion_lib_config
apply_inference_overrides = _BASE.apply_inference_overrides

_POSE_BANK_PATH = Path(
    os.environ.get(
        "HC_POSE_BANK_JSON",
        str(
            Path(__file__).resolve().parents[2]
            / "docs"
            / "artifacts"
            / "standing_pose_bank_v1.json"
        ),
    )
)
_CONTROLLED_DOFS = 33
_RESET_TILT = float(os.environ.get("HC_POSTURE_RESET_TILT", "0.0"))
# Reset momentum.  Every episode used to start from a dead stop: the reset set
# dof_pos, root_rot and height and left all velocities at zero, so the policy
# only ever saw "standing still in an odd stance" and never "already falling".
# Pushes were the sole source of momentum in the training distribution, which is
# why stage 1 was the only significant lever measured (30/64 -> 45/64, p~0.005,
# terminate_mean 42x lower).  Injecting the momentum at reset instead makes it a
# first-class part of every episode rather than an occasional 0.2 m/s nudge.
_RESET_VEL = float(os.environ.get("HC_FREE_RESET_VEL", "0.0"))
_RESET_ANG_VEL = float(os.environ.get("HC_FREE_RESET_ANGVEL", "0.0"))
_RESET_DOF_VEL = float(os.environ.get("HC_FREE_RESET_DOFVEL", "0.0"))
# The one ablated feature: arm B runs identically with this off, so a null result
# says the bottleneck is control authority rather than state estimation.
_BALANCE_OBS = os.environ.get("HC_FREE_BALANCE_OBS", "1") == "1"
# Contact *forces*, not the thresholded binary contacts already in the bundle:
# ankle strategy acts by moving the centre of pressure, which binary contact
# cannot express.
_FORCE_OBS = os.environ.get("HC_FREE_FORCE_OBS", "0") == "1"
# Capture point centred in the support base rather than merely over a foot.
_MARGIN_W = float(os.environ.get("HC_FREE_MARGIN_W", "0.0"))


# A locomotion bank additionally carries the velocity each mined frame actually
# had.  Standing banks do not (the pose is at rest by construction), so the
# fields are optional and default to zero -- v1/v2 banks load unchanged.
_BANK_USE_VEL = os.environ.get("HC_BANK_USE_VEL", "1") == "1"


def _load_bank():
    with _POSE_BANK_PATH.open() as f:
        poses = json.load(f)["poses"]
    dof = torch.tensor([p["reset_dof_pos"] for p in poses], dtype=torch.float32)
    rot = torch.tensor(
        [p["reset_root_rotation_xyzw"] for p in poses], dtype=torch.float32
    )
    height = torch.tensor(
        [p["reset_root_height"] for p in poses], dtype=torch.float32
    )
    zeros3 = [0.0, 0.0, 0.0]
    vel = torch.tensor(
        [p.get("reset_root_vel", zeros3) for p in poses], dtype=torch.float32
    )
    ang_vel = torch.tensor(
        [p.get("reset_root_ang_vel", zeros3) for p in poses], dtype=torch.float32
    )
    dof_vel = torch.tensor(
        [p.get("reset_dof_vel", [0.0] * dof.shape[1]) for p in poses],
        dtype=torch.float32,
    )
    return dof, rot, height, vel, ang_vel, dof_vel


_BANK_DOF, _BANK_ROT, _BANK_HEIGHT, _BANK_VEL, _BANK_ANGVEL, _BANK_DOFVEL = _load_bank()
_BANK_HAS_VEL = bool(
    _BANK_USE_VEL
    and (
        _BANK_VEL.abs().sum() > 0
        or _BANK_ANGVEL.abs().sum() > 0
        or _BANK_DOFVEL.abs().sum() > 0
    )
)


def _quat_mul_xyzw(a, b):
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        dim=-1,
    )


_BASE_ENV_RESET = BaseEnv.reset


def _posture_bank_reset(
    self,
    env_ids=None,
    sample_flat=False,
    force_default_mask=None,
    disable_motion_resample=False,
):
    """Reset every env to a randomly drawn standing stance from the bank.

    The upstream steering task resets from random reference-motion frames, which
    would hand ⑥ mid-stride states it has no way to interpret yet; the anchored
    experiment replaced that with one fixed pose, which is the limitation this
    experiment exists to remove.  Drawing from the bank keeps every reset a
    *valid, distinct* standing stance.
    """
    if force_default_mask is None:
        count = self.num_envs if env_ids is None else len(env_ids)
        force_default_mask = torch.ones(count, dtype=torch.bool, device=self.device)

    device = self.default_reset_state.dof_pos.device
    dtype = self.default_reset_state.dof_pos.dtype
    n = self.default_reset_state.dof_pos.shape[0]
    pick = torch.randint(_BANK_DOF.shape[0], (n,), device="cpu")

    dof = _BANK_DOF[pick].to(device=device, dtype=dtype)
    rot = _BANK_ROT[pick].to(device=device, dtype=dtype)
    height = _BANK_HEIGHT[pick].to(device=device, dtype=dtype)

    if _RESET_TILT > 0.0:
        angle = torch.randn(n, device=device, dtype=dtype).abs() * _RESET_TILT
        azimuth = torch.rand(n, device=device, dtype=dtype) * 2.0 * torch.pi
        axis = torch.stack(
            (torch.cos(azimuth), torch.sin(azimuth), torch.zeros_like(azimuth)),
            dim=-1,
        )
        half = angle * 0.5
        delta = torch.cat(
            (axis * torch.sin(half).unsqueeze(-1), torch.cos(half).unsqueeze(-1)),
            dim=-1,
        )
        rot = _quat_mul_xyzw(delta, rot)

    self.default_reset_state.dof_pos[:] = dof
    self.default_reset_state.root_rot[:] = rot
    self.default_reset_state.root_pos[:, 2] = height

    # A mid-stride pose spawned at a dead stop is not a locomotion state -- the
    # momentum is what makes it one, and half-normal noise is a poor substitute
    # for the momentum that actually belongs to the frame.  Lay the bank's own
    # velocities down first; the random knobs below then add on top, so a bank
    # without velocities reproduces the previous behaviour exactly.
    if _BANK_HAS_VEL:
        if self.default_reset_state.root_vel is not None:
            self.default_reset_state.root_vel[:] = _BANK_VEL[pick].to(
                device=device, dtype=dtype
            )
        if self.default_reset_state.root_ang_vel is not None:
            self.default_reset_state.root_ang_vel[:] = _BANK_ANGVEL[pick].to(
                device=device, dtype=dtype
            )
        if self.default_reset_state.dof_vel is not None:
            self.default_reset_state.dof_vel[:] = _BANK_DOFVEL[pick].to(
                device=device, dtype=dtype
            )
    else:
        if self.default_reset_state.root_vel is not None:
            self.default_reset_state.root_vel[:] = 0.0
        if self.default_reset_state.root_ang_vel is not None:
            self.default_reset_state.root_ang_vel[:] = 0.0
        if self.default_reset_state.dof_vel is not None:
            self.default_reset_state.dof_vel[:] = 0.0

    # Horizontal launch direction is uniform on the circle so no heading is
    # privileged; magnitude is half-normal so most resets stay gentle and the
    # hard ones are rare, which keeps the early curriculum learnable.
    if _RESET_VEL > 0.0 and self.default_reset_state.root_vel is not None:
        azimuth = torch.rand(n, device=device, dtype=dtype) * 2.0 * torch.pi
        speed = torch.randn(n, device=device, dtype=dtype).abs() * _RESET_VEL
        self.default_reset_state.root_vel[:, 0] += speed * torch.cos(azimuth)
        self.default_reset_state.root_vel[:, 1] += speed * torch.sin(azimuth)
    if _RESET_ANG_VEL > 0.0 and self.default_reset_state.root_ang_vel is not None:
        self.default_reset_state.root_ang_vel[:] += (
            torch.randn(n, 3, device=device, dtype=dtype) * _RESET_ANG_VEL
        )
    if _RESET_DOF_VEL > 0.0 and self.default_reset_state.dof_vel is not None:
        self.default_reset_state.dof_vel[:] += (
            torch.randn_like(self.default_reset_state.dof_vel) * _RESET_DOF_VEL
        )
    # Zero action must hold the stance the robot was handed, not some global
    # neutral -- otherwise the policy spends its whole budget dragging every
    # stance back to one pose, which is exactly the anchored behaviour.
    self.config.action_config["pd_action_offset"] = dof.clone()

    return _BASE_ENV_RESET(
        self,
        env_ids=env_ids,
        sample_flat=sample_flat,
        force_default_mask=force_default_mask,
        disable_motion_resample=disable_motion_resample,
    )


if not getattr(BaseEnv.reset, "_hc_posture_bank_reset", False):
    _posture_bank_reset._hc_posture_bank_reset = True
    BaseEnv.reset = _posture_bank_reset


def terrain_config(args):
    cfg = _BASE.terrain_config(args)
    cfg.spacing_between_scenes = 1.0
    return cfg


def env_config(robot_cfg, args):
    cfg = _BASE.env_config(robot_cfg, args)
    # Overwritten per reset; this only sets a sane value for the first frame.
    cfg.action_config["pd_action_offset"] = _BANK_DOF[0].clone().to(
        robot_cfg.default_dof_pos.device
    )
    cfg.control_components["steering"] = SteeringControlConfig(
        tar_speed_min=0.0,
        tar_speed_max=0.0,
        stop_probability=1.0,
        enable_rand_facing=False,
        heading_change_steps_min=100000,
        heading_change_steps_max=100001,
    )
    cfg.observation_components["posture_free_obs"] = MdpComponent(
        compute_func=compute_free_posture_observation,
        dynamic_vars={
            "root_rot": EnvContext.current.root_rot,
            "root_vel": EnvContext.current.root_vel,
            "root_ang_vel": EnvContext.current.root_ang_vel,
            "root_pos": EnvContext.current.root_pos,
            "ground_height": EnvContext.ground_heights,
            "rigid_body_contacts": EnvContext.current.rigid_body_contacts,
            "rigid_body_pos": EnvContext.current.rigid_body_pos,
            "contact_force_magnitudes": EnvContext.current_contact_force_magnitudes,
            "dof_pos": EnvContext.current.dof_pos,
            "dof_vel": EnvContext.current.dof_vel,
            "previous_processed_action": EnvContext.previous_processed_action,
        },
        static_params={
            "include_balance": _BALANCE_OBS,
            "include_force": _FORCE_OBS,
        },
    )
    cfg.reward_components = {
        "pelvis_height": MdpComponent(
            compute_func=compute_pelvis_height_reward,
            dynamic_vars={
                "root_pos": EnvContext.current.root_pos,
                "ground_height": EnvContext.ground_heights,
            },
            static_params={
                "target_height": float(
                    os.environ.get("HC_FREE_TARGET_HEIGHT", "0.93")
                ),
                # Tolerant on purpose: walking dips the pelvis every step.
                "coefficient": float(os.environ.get("HC_FREE_HEIGHT_COEF", "12.0")),
                "weight": float(os.environ.get("HC_FREE_HEIGHT_W", "0.20")),
            },
        ),
        "com_stability": MdpComponent(
            compute_func=compute_com_stability_reward,
            dynamic_vars={
                "root_rot": EnvContext.current.root_rot,
                "root_vel": EnvContext.current.root_vel,
                "root_ang_vel": EnvContext.current.root_ang_vel,
                "rigid_body_contacts": EnvContext.current.rigid_body_contacts,
            },
            static_params={
                "lin_vel_coef": float(os.environ.get("HC_FREE_LIN_VEL_COEF", "1.0")),
                "ang_vel_coef": float(os.environ.get("HC_FREE_ANG_VEL_COEF", "0.1")),
                "stable_speed": float(os.environ.get("HC_FREE_STABLE_SPEED", "0.25")),
                "weight": float(os.environ.get("HC_FREE_COM_W", "0.25")),
            },
        ),
        "capture_point": MdpComponent(
            compute_func=compute_capture_point_reward,
            dynamic_vars={
                "root_pos": EnvContext.current.root_pos,
                "root_vel": EnvContext.current.root_vel,
                "rigid_body_pos": EnvContext.current.rigid_body_pos,
                "rigid_body_contacts": EnvContext.current.rigid_body_contacts,
            },
            static_params={
                "tau": float(os.environ.get("HC_FREE_CAPTURE_TAU", "0.32")),
                "coefficient": float(os.environ.get("HC_FREE_CAPTURE_COEF", "10.0")),
                "weight": float(os.environ.get("HC_FREE_CAPTURE_W", "0.20")),
            },
        ),
        # Off by default: enabling it rebalances the other weights, so the
        # comparison must be run deliberately rather than inherited.
        **(
            {
                "support_margin": MdpComponent(
                    compute_func=compute_support_margin_reward,
                    dynamic_vars={
                        "root_pos": EnvContext.current.root_pos,
                        "root_vel": EnvContext.current.root_vel,
                        "rigid_body_pos": EnvContext.current.rigid_body_pos,
                        "rigid_body_contacts": EnvContext.current.rigid_body_contacts,
                    },
                    static_params={
                        "tau": float(os.environ.get("HC_FREE_CAPTURE_TAU", "0.32")),
                        "coefficient": float(
                            os.environ.get("HC_FREE_MARGIN_COEF", "8.0")
                        ),
                        "weight": _MARGIN_W,
                    },
                )
            }
            if _MARGIN_W > 0.0
            else {}
        ),
        # Restores the anchored recipe's three *pose-free* regularizers.  Dropping
        # them along with the anchor is what left the vibration termination
        # carrying cleanliness alone.
        "motion_cleanliness": MdpComponent(
            compute_func=compute_motion_cleanliness_reward,
            dynamic_vars={
                "root_ang_vel": EnvContext.current.root_ang_vel,
                "dof_vel": EnvContext.current.dof_vel,
                "current_processed_action": EnvContext.current_processed_action,
                "previous_processed_action": EnvContext.previous_processed_action,
            },
            static_params={
                "action_rate_coef": float(
                    os.environ.get("HC_FREE_ACTION_RATE_COEF", "10.0")
                ),
                "weight": float(os.environ.get("HC_FREE_CLEAN_W", "0.35")),
            },
        ),
    }
    cfg.termination_components = {
        "fall": MdpComponent(
            compute_func=compute_posture_fall_termination,
            dynamic_vars={
                "root_pos": EnvContext.current.root_pos,
                "ground_height": EnvContext.ground_heights,
            },
            static_params={"min_root_height": 0.55},
        )
    }
    # Without the nominal-pose reward there is nothing left implicitly discouraging
    # chatter, so the termination that solved it once is not optional here.
    _vib_limit = float(os.environ.get("HC_POSTURE_VIB_TERM_RMS", "3.5"))
    if _vib_limit > 0:
        cfg.termination_components["vibration_limit"] = MdpComponent(
            compute_func=compute_posture_vibration_termination,
            dynamic_vars={"dof_vel": EnvContext.current.dof_vel},
            static_params={"max_joint_speed_rms": _vib_limit},
        )
    # No crouch floor: a height termination is right for standing and wrong for
    # walking, where the pelvis dips every step.  Height is carried by the reward
    # instead, where it can be traded against balance.
    cfg.max_episode_length = int(os.environ.get("HC_POSTURE_EPISODE_LEN", "240"))
    return cfg


def configure_robot_and_simulator(robot_cfg, simulator_cfg, args):
    # Only sets the first-frame default; every reset overwrites from the bank.
    robot_cfg.default_dof_pos[:] = robot_cfg.default_dof_pos.new_tensor(
        _BANK_DOF[0].tolist()
    )
    robot_cfg.default_root_height = float(_BANK_HEIGHT[0].item())
    stage = int(os.environ.get("HC_POSTURE_CURRICULUM_STAGE", "0"))
    settings = {
        0: ((100000.0, 100001.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        1: ((2.0, 3.5), (0.20, 0.20, 0.0), (0.15, 0.15, 0.08)),
        2: ((1.25, 2.5), (0.35, 0.35, 0.0), (0.30, 0.30, 0.15)),
    }
    if stage not in settings:
        raise ValueError("HC_POSTURE_CURRICULUM_STAGE must be 0, 1, or 2")
    interval, linear, angular = settings[stage]
    simulator_cfg.domain_randomization = DomainRandomizationConfig(
        push=PushDomainRandomizationConfig(
            push_interval_range=interval,
            max_linear_velocity=linear,
            max_angular_velocity=angular,
        )
    )


def agent_config(robot_cfg, env_cfg, args):
    cfg = _BASE.agent_config(robot_cfg, env_cfg, args)
    if "posture_free_obs" not in cfg.model.in_keys:
        cfg.model.in_keys.append("posture_free_obs")
    cfg.model.actor.in_keys = ["posture_free_obs"]
    cfg.model.actor.mu_model.in_keys = ["posture_free_obs"]
    cfg.model.actor.mu_model._target_ = (
        "human_controller.models.posture_stabilized_motion_driver."
        "FreePostureFoundationPolicy"
    )
    num_dof = int(robot_cfg.default_dof_pos.numel())
    # gravity(3) + lin_vel(3) + ang_vel(3) + height(1) + contacts(4)
    # + balance(13, optional) + force(7, optional) + 3 x DOF
    obs_dim = (
        14
        + (BALANCE_OBS_DIM if _BALANCE_OBS else 0)
        + (FORCE_OBS_DIM if _FORCE_OBS else 0)
        + 3 * num_dof
    )
    os.environ["HC_FREE_OBS_DIM"] = str(obs_dim)
    try:
        cfg.model.actor.mu_model.posture_obs_dim = obs_dim
    except Exception:
        pass
    cfg.model.actor.actor_logstd = float(
        os.environ.get("HC_POSTURE_ACTOR_LOGSTD", "-2.5")
    )
    # 5e-5 with a single mini-epoch left clip_frac pinned at exactly 0 on the
    # anchored runs -- PPO's trust region was never reached.  Do not inherit it.
    cfg.model.actor_optimizer.lr = float(os.environ.get("HC_POSTURE_ACTOR_LR", "3e-4"))
    cfg.num_mini_epochs = int(os.environ.get("HC_POSTURE_MINI_EPOCHS", "4"))
    cfg.entropy_coef = float(
        os.environ.get(
            "HC_POSTURE_ENTROPY_COEF", str(getattr(cfg, "entropy_coef", 0.005))
        )
    )
    # No AMP.  Resets span many stances, so there is no single clip to imitate,
    # and a walking clip would push ⑥ toward being a second locomotion planner.
    # AMP off by default for the reasons above, but exposed: it is the only
    # remaining explanation for the anchored policy's lower vibration, and that
    # hypothesis is only testable if the weight can be raised.
    _amp_w = float(os.environ.get("HC_FREE_AMP_W", "0.0"))
    cfg.task_reward_w = 1.0 - _amp_w
    cfg.amp_parameters.discriminator_reward_w = _amp_w
    cfg.amp_parameters.discriminator_reward_threshold = 0.0
    cfg.amp_parameters.discriminator_max_cumulative_bad_transitions = 1_000_000_000
    if getattr(cfg, "evaluator", None) is not None:
        cfg.evaluator.evaluation_components = {}
        try:
            cfg.evaluator.eval_metrics_every = None
        except Exception:
            pass
    return cfg
