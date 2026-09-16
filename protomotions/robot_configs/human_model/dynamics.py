# SPDX-License-Identifier: Apache-2.0
"""Active strength envelope and conservative/dissipative passive joint forces.

All tensors use common DOF order and SI units (rad, rad/s, Nm). The original
controller's PD gains are deliberately not used as passive tissue stiffness.
"""

import math
import json
from pathlib import Path

import torch

from .profile import validate_profile


def fatigue_compartment_step(state, target, *, fatigue_rate, recovery_rate, dt, tracking_rate=10.0, rest_multiplier=1.0):
    """Candidate 3-compartment source ODE; state order is resting/active/fatigued.

    Fractions sum to one; rates are per second. Frey-Law2012 Appendix Eq1-6
    supplies this recruitment/decay law, not a locomotion controller. This
    numerical component is used by the opt-in HumanJointModel fatigue path.
    SSP RK2 with bounded Euler substeps preserves nonnegative compartments.
    Optional Looft2018 rest augmentation applies only at target == 0 and to
    both sides of fatigued-to-resting transfer. Default 1 retains original 3CC.
    """
    if not state.is_floating_point() or state.ndim < 1 or state.shape[-1] != 3:
        raise ValueError('floating state [...,3] required')
    target = torch.as_tensor(target,device=state.device,dtype=state.dtype)
    if target.shape != state.shape[:-1]:
        raise ValueError('target must match the state batch shape')
    if (not torch.isfinite(state).all() or (state < 0).any()
            or not torch.allclose(state.sum(-1),torch.ones_like(target),atol=1e-6,rtol=0)
            or not torch.isfinite(target).all() or (target < 0).any() or (target > 1).any()):
        raise ValueError('unit-sum nonnegative state and target in [0,1] required')
    if (not all(math.isfinite(x) for x in (fatigue_rate,recovery_rate,tracking_rate,dt,rest_multiplier))
            or min(fatigue_rate,recovery_rate) < 0 or tracking_rate <= 0 or dt <= 0 or rest_multiplier < 1):
        raise ValueError('nonnegative finite fatigue/recovery, positive tracking rate/dt and rest multiplier >= 1 required')
    effective_recovery = recovery_rate*torch.where(target == 0, torch.full_like(target, rest_multiplier), torch.ones_like(target))
    count = max(1,math.ceil(dt*(tracking_rate+fatigue_rate+recovery_rate*rest_multiplier)/.9))
    h = dt/count

    def flow(value):
        resting,active,fatigued = value.unbind(-1)
        difference = target-active
        recruitment = tracking_rate*torch.where(difference > 0,
                                                 torch.minimum(difference,resting),difference)
        fatigue = fatigue_rate*active
        recovery = effective_recovery*fatigued
        return torch.stack((-recruitment+recovery,recruitment-fatigue,fatigue-recovery),-1)

    result = state
    for _ in range(count):
        first = result+h*flow(result)
        result = .5*result+.5*(first+h*flow(first))
    # Remove only floating-point mass drift; do not clip negative states.
    return result/result.sum(-1,keepdim=True)


class HumanJointModel:
    def __init__(self, profile, dof_names, device="cpu", dtype=torch.float32, *, features=(), strength_cohort='equal_sex', strength_reference_size=None, active_strength_scale=None, fatigue_regions=None, fatigue_rest_multiplier=1.0):
        validate_profile(profile)
        self.features = frozenset(features)
        if (not math.isfinite(fatigue_rest_multiplier) or fatigue_rest_multiplier < 1
                or (fatigue_rest_multiplier != 1 and "fatigue" not in self.features)):
            raise ValueError("fatigue rest multiplier >= 1 requires fatigue when nondefault")
        self.fatigue_rest_multiplier = float(fatigue_rest_multiplier)
        if {'activation','fatigue'} <= self.features:
            raise ValueError('fatigue includes recruitment dynamics; do not combine with activation')
        if bool(fatigue_regions) != ('fatigue' in self.features):
            raise ValueError('fatigue requires an explicit nonempty fatigue_regions mapping')
        if strength_reference_size is not None:
            if 'strength' not in self.features or len(strength_reference_size) != 2:
                raise ValueError('strength_reference_size requires strength and (mass_kg, height_m)')
            strength_reference_size = tuple(float(x) for x in strength_reference_size)
            if not all(math.isfinite(x) and x > 0 for x in strength_reference_size):
                raise ValueError('strength reference mass and height must be positive finite values')
        self.strength_reference_size = strength_reference_size
        if strength_cohort not in ('equal_sex','male','female'):
            raise ValueError('strength_cohort must be equal_sex, male, or female')
        if strength_cohort != 'equal_sex' and 'strength' not in self.features:
            raise ValueError('source cohort selection requires strength')
        self.strength_cohort = strength_cohort
        if self.features - {"strength", "activation", "passive_fit", "strength_coupling", "wrist_damping", "fatigue"}:
            raise ValueError(f"Unknown human-model candidate features: {self.features}")
        if 'strength_coupling' in self.features and 'strength' not in self.features:
            raise ValueError('strength_coupling requires strength')
        if self.features:
            self.candidate_parameters = json.loads(Path(__file__).with_name("candidate_parameters.json").read_text())
        if set(dof_names) != set(profile["joints"]) or len(set(dof_names)) != len(dof_names):
            raise ValueError("Human profile requires the exact 69-axis SMPL joint set")
        self.names = list(dof_names)
        self.fatigue_regions = dict(fatigue_regions or {})
        self._fatigue = None
        self._fatigue_groups = []
        self._fatigue_mask = torch.zeros(len(self.names),device=device,dtype=torch.bool)
        if self.fatigue_regions:
            regions = self.candidate_parameters['fatigue']['regions']
            if any(name not in self.names or region not in regions for name,region in self.fatigue_regions.items()):
                raise ValueError('fatigue_regions requires known DOFs and source regions')
            for region in sorted(set(self.fatigue_regions.values())):
                indices = [self.names.index(name) for name,r in self.fatigue_regions.items() if r == region]
                self._fatigue_mask[indices] = True
                self._fatigue_groups.append((indices,regions[region]))
        self.profile_id = profile["id"]
        rows = [profile["joints"][n] for n in self.names]

        def tensor(values):
            return torch.tensor(values, device=device, dtype=dtype)

        # Anderson source rows are young male then young female. This selects
        # source curves only: body mass/inertia and other joint data do not change.
        self.strength_cohort_weights = tensor({'equal_sex':[.5,.5], 'male':[1.,0.], 'female':[0.,1.]}[strength_cohort])
        self.lower = tensor([r["rom_deg"][0] for r in rows]) * math.pi / 180
        self.upper = tensor([r["rom_deg"][1] for r in rows]) * math.pi / 180
        self.negative = tensor([r["active_nm"][0] for r in rows])
        self.positive = tensor([r["active_nm"][1] for r in rows])
        self.active_strength_scale = torch.ones((len(self.names),2),device=device,dtype=dtype)
        for name, values in (active_strength_scale or {}).items():
            if name not in self.names or len(values)!=2 or not all(math.isfinite(v) and v>=0 for v in values):
                raise ValueError('active_strength_scale requires known DOFs and finite nonnegative [negative, positive] factors')
            self.active_strength_scale[self.names.index(name)] = tensor(values)
        self.negative *= self.active_strength_scale[:,0]
        self.positive *= self.active_strength_scale[:,1]
        self.rest = tensor([r["rest_deg"] for r in rows]) * math.pi / 180
        self.k_negative = tensor([r["stiffness_nm_per_rad"][0] for r in rows])
        self.k_positive = tensor([r["stiffness_nm_per_rad"][1] for r in rows])
        self.damping = tensor([r["damping_nms_per_rad"] for r in rows])
        if 'wrist_damping' in self.features:
            wrist = self.candidate_parameters['wrist_damping']
            # VI is a slope versus cycles/s. A velocity damper's quadrature
            # torque/angle amplitude is B * 2*pi*f, hence B = VI/(2*pi).
            slope = .5 * (wrist['reported_vi_female'] + wrist['reported_vi_male'])
            for name in wrist['joint_names']:
                self.damping[self.names.index(name)] = slope / (2 * math.pi)
        components = (self.candidate_parameters["passive_fit"]["passive_exponentials"]
                      if "passive_fit" in self.features else profile["passive_exponentials"])
        self.passive_component_names = [c['name'] for c in components]
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
        self._activation = None
        self._strength_rows = []
        if self.features:
            params = self.candidate_parameters
            self.activation_rise_s = params["activation"]["rise_s"]
            self.activation_fall_s = params["activation"]["fall_s"]
            if "strength" in self.features:
                strength = params["strength"]
                scales = tensor([r["mass_kg"] * r["height_m"] * strength["gravity_m_s2"] for r in strength["cohorts"]])
                if self.strength_reference_size is not None:
                    mass,height = self.strength_reference_size
                    scales.fill_(mass*height*strength['gravity_m_s2'])
                for row in strength["directions"].values():
                    coefficients = tensor(row["coefficients"])
                    coefficients[:, 0] *= scales
                    for side in ("L", "R"):
                        i = self.names.index(f"{side}_{row['joint']}")
                        self._strength_rows.append((i, row, coefficients))
                        # The concentric rational function <= 1 for these data.
                        bound = (self.strength_cohort_weights * coefficients[:, 0] * (1 + coefficients[:, 5] * row["eccentric_limit_rad_s"])).sum()
                        bound *= self.active_strength_scale[i, int(row['sign']>0)]
                        if 'strength_coupling' in self.features and row['joint']=='Ankle_y' and row['sign']>0:
                            coupling=params['strength_coupling']
                            bound*=math.exp(-coupling['beta_per_rad']*(coupling['knee_domain_rad'][0]-coupling['reference_knee_rad']))
                        previous = torch.maximum(self.negative[i], self.positive[i])
                        self.backend_limit[i] += (bound - previous).clamp_min(0)

    def strength_caps(self, q, qd, *, directional_domain=False):
        """Caps and extrapolation flags for enabled candidates; stateless.

        Optional flags have shape [..., DOF, 2] in negative/positive torque
        order. Default flags retain the conservative OR across directions.
        A false flag does not establish biological coverage of omitted features.
        """
        negative = self.negative.expand_as(q).clone()
        positive = self.positive.expand_as(q).clone()
        outside = (torch.zeros((*q.shape, 2), device=q.device, dtype=torch.bool)
                   if directional_domain else torch.zeros_like(q, dtype=torch.bool))
        for i, row, c in self._strength_rows:
            angle = q[..., i] * row["clinical_angle_sign"]
            velocity = qd[..., i] * row["sign"]  # positive mechanical work = concentric
            lo, hi = row["angle_domain_rad"]
            vmin, vmax = -row["eccentric_limit_rad_s"], row["concentric_limit_rad_s"]
            flag = (angle < lo) | (angle > hi) | (velocity < vmin) | (velocity > vmax)
            if directional_domain:
                outside[..., i, int(row['sign'] > 0)] |= flag
            else:
                outside[..., i] |= flag
            angle = angle.clamp(lo, hi).unsqueeze(-1)
            velocity = velocity.clamp(vmin, vmax).unsqueeze(-1)
            speed = velocity.abs()
            numerator = 2*c[:, 3]*c[:, 4] + speed*(c[:, 4] - 3*c[:, 3])
            denominator = 2*c[:, 3]*c[:, 4] + speed*(2*c[:, 4] - 4*c[:, 3])
            magnitude = c[:, 0] * torch.cos(c[:, 1]*(angle - c[:, 2])).clamp_min(0)
            magnitude *= (numerator / denominator).clamp_min(0)
            magnitude *= 1 + c[:, 5] * (-velocity).clamp_min(0)
            target = positive if row["sign"] > 0 else negative
            target[..., i] = (magnitude * self.strength_cohort_weights).sum(-1) * self.active_strength_scale[i,int(row['sign']>0)]
        if 'strength_coupling' in self.features:
            coupling=self.candidate_parameters['strength_coupling']
            lo,hi=coupling['knee_domain_rad']
            for side in ('L','R'):
                knee=self.names.index(f'{side}_Knee_y');ankle=self.names.index(f'{side}_Ankle_y')
                angle=q[...,knee]
                positive[...,ankle]*=torch.exp(-coupling['beta_per_rad']*(angle.clamp(lo,hi)-coupling['reference_knee_rad']))
                # Identified only at isometric conditions; dynamic separability
                # is an explicit candidate assumption, not measured validation.
                flag=(angle<lo)|(angle>hi)|(qd[...,ankle].abs()>1e-6)|(qd[...,knee].abs()>1e-6)
                if directional_domain:
                    outside[...,ankle,1] |= flag  # Only plantarflexion uses this coupling.
                else:
                    outside[...,ankle] |= flag
        return negative, positive, outside

    def reset(self, env_ids=None):
        """Clear candidate activation at episode boundaries, including partial reset."""
        if self._activation is not None:
            if env_ids is None:
                self._activation.zero_()
            else:
                self._activation[env_ids] = 0
        if self._fatigue is not None:
            if env_ids is None:
                self._fatigue.zero_()
                self._fatigue[...,0] = 1
            else:
                self._fatigue[env_ids] = 0
                self._fatigue[env_ids,...,0] = 1

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

    def passive_guard_flags(self, q, qd):
        """Report force-affecting numerical guard use; not anatomical validity.

        Call for an assay/recorder as needed; no extra work in the force loop.
        Exponential flags are per coupled energy component, others per DOF.
        """
        if q.shape != qd.shape or q.shape[-1] != len(self.names):
            raise ValueError('q and qd must share [..., dof] shape')
        displacement=q-self.rest
        k=torch.where(displacement<0,self.k_negative,self.k_positive)
        return {
            'exponential':((q@self.a.T+self.offset)>=math.log(self.exp_cap))&(self.energy_scale>0),
            'spring':(k*displacement).abs()>=self.spring_cap,
            'damping':(self.damping*qd).abs()>=self.damping_cap,
        }

    def passive_impedance(self, q, qd):
        """Local tissue tangent: delta_tau = -K @ delta_q - B * delta_qd.

        K includes off-diagonal coupling; B is diagonal. Neither includes engine
        limit/contact reactions, controller PD gains, or active co-contraction.
        At piecewise-law corners the returned branch derivative is diagnostic
        only: differentiable is false. This is a model prediction, not a measured
        human impedance or evidence that unimplemented resistance is zero.
        """
        if q.shape != qd.shape or q.shape[-1] != len(self.names):
            raise ValueError('q and qd must share [..., dof] shape')
        z = q @ self.a.T + self.offset
        threshold = math.log(self.exp_cap)
        weight = self.energy_scale * torch.exp(z.clamp(max=threshold)) * (z < threshold)
        stiffness = torch.einsum('...c,ci,cj->...ij', weight, self.a, self.a)
        displacement = q - self.rest
        k = torch.where(displacement < 0, self.k_negative, self.k_positive)
        spring_load = k * displacement
        stiffness = stiffness + torch.diag_embed(k * (spring_load.abs() < self.spring_cap))
        damping_load = self.damping * qd
        damping = self.damping.expand_as(qd) * (damping_load.abs() < self.damping_cap)
        tolerance = 32 * torch.finfo(q.dtype).eps
        corners = ((z - threshold).abs() <= tolerance * (1 + abs(threshold))).any(-1)
        corners |= ((displacement.abs() <= tolerance) & (self.k_negative != self.k_positive)).any(-1)
        corners |= ((spring_load.abs() - self.spring_cap).abs() <= tolerance * (1 + self.spring_cap)).any(-1)
        corners |= ((damping_load.abs() - self.damping_cap).abs() <= tolerance * (1 + self.damping_cap)).any(-1)
        return {'stiffness_nm_per_rad': stiffness, 'damping_nms_per_rad': damping,
                'differentiable': ~corners}

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

    def torques(self, requested, q, qd, *, dt=None, coactivation=None):
        """Return total, active and passive separately; strength caps active only.

        Optional coactivation [..., DOF] in [0,1] recruits an equal opposing
        torque from the remaining directional capacity. It requires activation
        or a mapped fatigue axis. With fatigue, strength_caps remains the fresh
        envelope; the physical-step diagnostic caps include fatigue reduction.
        This joint-level allocation is not muscle force or a calibrated active
        stiffness law. Omission preserves the original net-command behavior.
        """
        return self._torque_components(requested, q, qd, dt=dt, coactivation=coactivation)[:3]

    def _torque_components(self, requested, q, qd, *, dt=None, coactivation=None):
        """Compute a single physical step and retain its exact diagnostic terms."""
        if requested.shape != q.shape or qd.shape != q.shape or q.shape[-1] != len(self.names):
            raise ValueError("requested, q and qd must share [..., dof] shape")
        if coactivation is not None:
            if not {'activation','fatigue'} & self.features:
                raise ValueError('coactivation requires an activation or fatigue candidate')
            coactivation = torch.as_tensor(coactivation, device=q.device, dtype=q.dtype)
            if (coactivation.shape != q.shape or not torch.isfinite(coactivation).all()
                    or (coactivation < 0).any() or (coactivation > 1).any()):
                raise ValueError('coactivation must have matching [..., dof] shape and finite values in [0,1]')
            if 'fatigue' in self.features and (coactivation[...,~self._fatigue_mask] != 0).any():
                raise ValueError('coactivation under fatigue requires mapped axes')
        negative, positive, outside = self.strength_caps(q, qd)
        active = torch.maximum(torch.minimum(requested, positive), -negative)
        if {'activation','fatigue'} & self.features:
            if dt is None or not math.isfinite(dt) or dt <= 0:
                raise ValueError("activation requires positive physical-substep dt")
            if 'activation' in self.features and self._activation is None:
                self._activation = torch.zeros((*q.shape, 2), device=q.device, dtype=q.dtype)
            if 'activation' in self.features and self._activation.shape != (*q.shape, 2):
                raise ValueError("activation batch changed; use a new model instance")
            caps = torch.stack((negative, positive), -1)
            effort = torch.stack(((-active).clamp_min(0), active.clamp_min(0)), -1)
            if coactivation is not None:
                # Equal Nm, not equal normalized activation: asymmetric caps
                # must not create an unintended steady net torque.
                shared_headroom = (caps - effort).amin(-1).clamp_min(0)
                effort = effort + (coactivation * shared_headroom).unsqueeze(-1)
            target = torch.where(caps > 0, effort / caps.clamp_min(torch.finfo(q.dtype).eps), 0)
            if 'fatigue' in self.features:
                if self._fatigue is None:
                    self._fatigue = torch.zeros((*q.shape,2,3),device=q.device,dtype=q.dtype)
                    self._fatigue[...,0] = 1
                if self._fatigue.shape != (*q.shape,2,3):
                    raise ValueError('fatigue batch changed; use a new model instance')
                updated = self._fatigue.clone()
                for indices,params in self._fatigue_groups:
                    updated[...,indices,:,:] = fatigue_compartment_step(
                        self._fatigue[...,indices,:,:],target[...,indices,:],dt=dt,
                        fatigue_rate=params['fatigue_rate_per_s'],recovery_rate=params['recovery_rate_per_s'],
                        tracking_rate=self.candidate_parameters['fatigue']['tracking_rate_per_s'],
                        rest_multiplier=self.fatigue_rest_multiplier)
                self._fatigue = updated
                fatigue_active = updated[...,1,1]*positive-updated[...,0,1]*negative
                active = torch.where(self._fatigue_mask,fatigue_active,active)
                negative = negative*(1-updated[...,0,2])
                positive = positive*(1-updated[...,1,2])
            else:
                tau = torch.where(target > self._activation,
                                  torch.full_like(target, self.activation_rise_s),
                                  torch.full_like(target, self.activation_fall_s))
                self._activation = target + (self._activation - target) * torch.exp(-dt / tau)
                active = self._activation[..., 1] * positive - self._activation[..., 0] * negative
        elastic = self.elastic_torque(q)
        damping = self.damping_torque(qd)
        passive = elastic + damping
        return active + passive, active, passive, elastic, damping, negative, positive, outside

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
        total, active, passive, elastic, damping, negative, positive, outside = self._torque_components(
            requested, state.dof_pos, state.dof_vel, dt=1.0 / simulator.config.sim.fps)
        simulator.human_requested_torques = requested
        simulator.human_active_torques = active
        simulator.human_passive_torques = passive
        simulator.human_elastic_torques = elastic
        simulator.human_damping_torques = damping
        simulator.human_applied_torques = total
        simulator.human_negative_caps, simulator.human_positive_caps, simulator.human_strength_outside_domain = negative, positive, outside
        # A recorder can consume every substep without retaining unbounded GPU tensors.
        callback = getattr(simulator, "human_model_trace_callback", None)
        if callback is not None:
            callback(simulator, state)
        simulator._apply_simulator_torques(total[..., simulator.data_conversion.dof_convert_to_sim])
