# SPDX-License-Identifier: Apache-2.0
"""Active strength envelope and conservative/dissipative passive joint forces.

All tensors use common DOF order and SI units (rad, rad/s, Nm). The original
controller's PD gains are deliberately not used as passive tissue stiffness.
"""

import math

import torch


class HumanJointModel:
    def __init__(self, profile, dof_names, device="cpu", dtype=torch.float32):
        if set(dof_names) != set(profile["joints"]) or len(set(dof_names)) != len(dof_names):
            raise ValueError("Human profile requires the exact 69-axis SMPL joint set")
        self.names = list(dof_names)
        self.profile_id = profile["id"]
        rows = [profile["joints"][n] for n in self.names]

        def tensor(values):
            return torch.tensor(values, device=device, dtype=dtype)

        self.lower = tensor([r["rom_deg"][0] for r in rows]) * math.pi / 180
        self.upper = tensor([r["rom_deg"][1] for r in rows]) * math.pi / 180
        self.negative = tensor([r["active_nm"][0] for r in rows])
        self.positive = tensor([r["active_nm"][1] for r in rows])
        self.rest = tensor([r["rest_deg"] for r in rows]) * math.pi / 180
        self.k_negative = tensor([r["stiffness_nm_per_rad"][0] for r in rows])
        self.k_positive = tensor([r["stiffness_nm_per_rad"][1] for r in rows])
        self.damping = tensor([r["damping_nms_per_rad"] for r in rows])
        components = profile["passive_exponentials"]
        self.a = tensor([[c["coeff_per_rad"].get(n, 0) for n in self.names] for c in components])
        self.offset = tensor([c["offset"] for c in components])
        self.energy_scale = tensor([c["energy_scale_j"] for c in components])
        numerical = profile["numerics"]
        # Tangent continuation of each convex energy keeps extrapolated forces
        # bounded WITHOUT clipping coupled joint torques independently.
        self.exp_cap = numerical["exponential_force_cap_nm"]
        self.spring_cap = numerical["linear_spring_force_cap_nm"]
        self.damping_cap = numerical["damping_force_cap_nm"]
        # For this parameterization energy_scale * primary beta == 1.
        component_bound = self.exp_cap * self.energy_scale[:, None] * self.a.abs()
        self.backend_limit = torch.maximum(self.negative, self.positive) + component_bound.sum(0)
        self.backend_limit += (self.k_negative + self.k_positive > 0) * self.spring_cap
        self.backend_limit += (self.damping > 0) * self.damping_cap
        # Engines need a positive motor ceiling even for near-locked passive axes.
        self.backend_limit = self.backend_limit.clamp_min(0.01)

    def limit_active(self, requested):
        return torch.maximum(torch.minimum(requested, self.positive), -self.negative)

    def elastic_torque(self, q):
        z = q @ self.a.T + self.offset
        f = torch.exp(z.clamp(max=math.log(self.exp_cap)))
        coupled = -(f * self.energy_scale) @ self.a
        displacement = q - self.rest
        k = torch.where(displacement < 0, self.k_negative, self.k_positive)
        spring = (k * displacement).clamp(-self.spring_cap, self.spring_cap)
        return coupled - spring

    def damping_torque(self, qd):
        # Saturation preserves dissipation: tau_d * qd <= 0 for every axis.
        return -(self.damping * qd).clamp(-self.damping_cap, self.damping_cap)

    def passive_torque(self, q, qd):
        return self.elastic_torque(q) + self.damping_torque(qd)

    def potential_energy(self, q):
        """Analytic energy for auditing passivity, including extrapolation guards."""
        z = q @ self.a.T + self.offset
        threshold = math.log(self.exp_cap)
        e = torch.exp(z.clamp(max=threshold)) + self.exp_cap * (z - threshold).clamp_min(0)
        displacement = q - self.rest
        k = torch.where(displacement < 0, self.k_negative, self.k_positive)
        transition = self.spring_cap / k.clamp_min(torch.finfo(q.dtype).eps)
        absolute = displacement.abs()
        limited = torch.minimum(absolute, transition)
        spring_e = 0.5 * k * limited.square() + self.spring_cap * (absolute - transition).clamp_min(0)
        return (e * self.energy_scale).sum(-1) + spring_e.sum(-1)

    def torques(self, requested, q, qd):
        """Return total, active and passive separately; strength caps active only."""
        active = self.limit_active(requested)
        passive = self.passive_torque(q, qd)
        return active + passive, active, passive

    def apply(self, simulator):
        from protomotions.robot_configs.base import ControlType

        command = simulator._common_actions
        randomization = simulator._domain_randomization
        if randomization is not None and "action_noise" in randomization:
            command = command.clone()
            noise = randomization["action_noise"]
            command[..., noise["dof_indices"]] += noise["action_noise"]
        state = simulator._get_simulator_dof_state().convert_to_common(simulator.data_conversion)
        # robot_config retains the checkpoint's action semantics. Only backend
        # drives switch to torque so active and passive can be added separately.
        mode = simulator.robot_config.control.control_type
        if mode in (ControlType.BUILT_IN_PD, ControlType.PROPORTIONAL):
            requested = simulator._common_p_gains * (command - state.dof_pos)
            requested -= simulator._common_d_gains * state.dof_vel
        elif mode == ControlType.TORQUE:
            requested = command
        else:
            raise ValueError(f"Unsupported human model command mode: {mode}")
        total, active, passive = self.torques(requested, state.dof_pos, state.dof_vel)
        simulator.human_active_torques = active
        simulator.human_passive_torques = passive
        simulator._apply_simulator_torques(total[..., simulator.data_conversion.dof_convert_to_sim])
