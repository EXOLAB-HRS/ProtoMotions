"""Variable joint impedance for the teacher (vic_tc_v1, candidate, 2026-10-22).

The policy keeps its 59 PD targets and gets 2 * J extra outputs for J selected joints:
a stiffness residual delta and a feed-forward torque residual tau_res. Per joint,

    k_p = exp(lk + delta),  k_d = exp(lb + delta),  delta = delta_max * tanh(a)
    tau = tau_ff + k_p * (q_ref - q) - k_d * qdot          (PhysX implicit drive + feed-forward)
    tau_ff = beta * tau_prior + lowpass(tau_res),  tau_res = tau_res_scale * tau_max * tanh(b)
    tau_prior = (k_old - k_p) * (q_ref - q) - (b_old - k_d) * qdot   (control step, held)

lk, lb follow a reference that moves in log space from the current gains (alpha = 0) to a human
reference (alpha = 1) chosen by each leg's state (swing / stance / both feet standing), smoothed
over gain_smooth_s. Human references are geometric centres of literature ranges
(html/HumanModel_v3_1_261022_joint_impedance_vs_human_ghlee.html); joints without a human value
(frontal) keep the current gain as reference. tau_prior lets the slow part of the torque match the
parent (r25) gains while the drive itself is already soft; beta goes 1 -> 0 so the policy takes the
torque over. alpha and beta advance on a step schedule and pause while the fall rate is above the
baseline measured at alpha = 0.

State is per leg from the cadence stance rule (foot body heights), not from a gait clock, so it also
applies to standing, stopping and turning.

Candidate options (2026-10-26, off by default): delta_band (delta_max per joint and leg state from the human range)
and delta_lowpass_s (lowpass on delta) or delta_tau_up_s / delta_tau_down_s (activation-like asymmetric lowpass);
recipe vic_TC_r11.

vic_tc_v2 (2026-10-28, user-approved version: stiffness / damping should not jitter like a trajectory but take clearly
separated, physiologically reasonable levels for swing / stance / standing / disturbance response). Per joint (hip,
knee, ankle, frontal hip, subtalar), the existing residual becomes a co-activation level inside the human range:

    u_j = tanh(a_j) in [-1, 1],  delta_j = h_j(state) * u_j,  h_j = half log-width of the human range of state

with delta_band_margin = 1 (u = -1 / 0 / +1 gives the low end / geometric centre / high end of the range, never
outside it; v1 allowed x1.5 beyond), an activation-like filter (delta_tau_up_s / delta_tau_down_s, 50 / 150 ms in the
v2 recipes; v1 15 / 50 ms) and, with coact_unit_cost, the cocontraction cost on the level, mean_j relu(u_j), so it means
the same for every joint. The filtered level u_j = delta_j / h_j is written to the gain log as "u".

vic_tc_v3 (2026-10-28, user-approved structure; vic_law = "v3", default "v2" keeps every earlier recipe unchanged).
v2 made k a lookup of the human reference by leg state (validation circular), the reference (perturbation-measured)
already contains reflex contributions, and the policy never moved u. v3 builds k from three terms, per joint j:

    k_p,j = k_base,j + c_j * |tau_j|_f * g_j(qdot) + k_co,j(u_j),      k_co,j = u_j * k_co_max,j,  u_j in [0, 1]
    k_d,j = d_j * sqrt(k_p,j)                                           (damping ratio fixed per joint, B ~ sqrt(k))
    k_p, k_d clamped to a wide hard corridor [k_lo_f * lo_s, k_hi_f * hi_s] of the HMV-ImpRef range of the leg state s

k_base: relaxed / low-activation intrinsic stiffness (swing-like, not the reference centre). c_j |tau_j|: load-dependent
intrinsic (short-range) stiffness, automatic, not learned; tau_j = the joint's active torque (PD + feed-forward, capped)
of the previous control step (simulator.human_active_torques, last substep; no algebraic loop), filtered with an
activation-like asymmetric lowpass (v3_load_tau_up_s / v3_load_tau_down_s). g_j = omega / (omega + |qdot|) when
v3_srs_omega > 0 (short-range stiffness yields during fast joint motion), else 1. k_co(u): co-contraction stiffness
learned through u (relu(tanh(a)) by default, so the parent's a ~ 0 means relaxed), filtered as in v2
(delta_tau_up_s / delta_tau_down_s). The band cost becomes a dead-zone guard: zero inside [lo / m, hi * m] of the
state's HMV-ImpRef range (m = v3_guard_margin), log2 distance outside; it never pulls toward the centre. The
cocontraction cost is either mean_j u_j ("unit") or the co-contraction torque share mean_j k_co,j / (c_j tau_max,j)
("torque": the agonist + antagonist torque that would give k_co through the same short-range law, i.e. the extra
muscle force that costs metabolic energy). Each term is written to the gain log (k_base, k_load, k_co, tau_load).
Parameters and sources: VIC3_PARAMS below.
"""
from dataclasses import dataclass, field
import atexit
import json
import math
import os

import torch
from torch import Tensor

from protomotions.envs.control.cadence_gait_prior import (
    CadenceGaitPercentSteering, CadenceGaitPercentSteeringConfig, steering_path)

VIC_BAND_COST = steering_path("vic_band_cost")
VIC_COCONTRACTION = steering_path("vic_cocontraction")
VIC_EFFORT = steering_path("vic_effort")
# vic_TC_r2 (2026-10-23): foot impact above the human step peak, and a metabolic proxy of the sagittal leg joints.
VIC_IMPACT = steering_path("vic_impact")
VIC_METAB = steering_path("vic_metab")

# Human joint impedance ranges (k N m/rad, b N m s/rad) for swing / stance / standing, model 63.3 kg.
# Hip and knee in stance and standing: relaxed-static values (no walking stance measurement).
# Ankle swing: assumption (low end of the walking range).
HUMAN_RANGES = {
    "Hip_y": {"k": ((34, 66), (50, 318), (50, 318)), "b": ((1.9, 4.6), (2, 21), (2, 21))},
    "Knee_y": {"k": ((0.5, 3.5), (15, 75), (15, 75)), "b": ((0.02, 0.14), (1, 2.2), (1, 2.2))},
    "Ankle_y": {"k": ((22, 100), (22, 406), (290, 340)), "b": ((0.6, 1.5), (0.6, 1.5), (0.6, 1.5))},
}
# Frontal joints (candidate, 2026-10-26, used only with frontal_human_ref): scale-level ranges, not phase references
# (share/HMV_v1_teacher_261026/human_impedance_ref_3state.csv, frontal rows).
# Hip_x stance: WBDS frontal hip quasi-stiffness (moment / angle slope, 0-50 % cycle, 753 cycles, p10-p90 167-679, median
#   335 N m/rad at 63.3 kg); standing: Winter 1998 quiet-stance M/L stiffness 627-1137 N m/rad shared by two hips;
#   swing: assumption = sagittal hip swing. Damping: assumption = sagittal hip.
# Subtalar_x (ankle inversion-eversion): relaxed 8.8 +- 2.2 N m/rad, 0.18 +- 0.09 N m s/rad (Ho, Lee, Hogan 2016);
#   active stance: relaxed + 1.1 N m/rad per % MVC (Lee, Krebs, Hogan 2014) at 10-50 % MVC, inside the WBDS frontal ankle
#   quasi-stiffness p10-p90 10-146. Damping above relaxed: assumption.
HUMAN_RANGES_FRONTAL = {
    "Hip_x": {"k": ((34, 66), (100, 700), (300, 600)), "b": ((1.9, 4.6), (2, 21), (2, 21))},
    "Subtalar_x": {"k": ((5, 15), (10, 70), (10, 30)), "b": ((0.1, 0.3), (0.2, 1.5), (0.2, 0.6))},
}


# vic_tc_v3 per-joint parameters (candidate, 2026-10-28). k_base N m/rad, c 1/rad (N m/rad per N m of active torque).
# Ankle_y c = 3.5 (medium confidence): standing intrinsic stiffness vs ankle torque, Sakanaka, Gill, Lakie, Reynolds 2018
#   (PLOS ONE 13:e0193850; +28 % mgh over +26 N m, ~3.3 /rad per leg), Mirbagheri, Barbeau, Kearney 2000 (+30 % stiffness
#   for 12 -> 24 N m), Loram & Lakie 2002 (5 -> 6 N m/deg over 5-25 N m, ~2.9 /rad); walking: Lee & Hogan 2015 and
#   Shorter & Rouse 2018 stance stiffness against the plantarflexion moment at the same phase give ~1.2-4.3 /rad;
#   Casadio 2005 higher (~7). Isometric slopes are higher than walking ones (Achilles tendon dominates above low load).
# Knee_y c = 2.5, Hip_y c = 2.0, Hip_x c = 2.5 (low confidence, no walking measurement of the slope): muscle short-range
#   stiffness k = gamma F / l0 (gamma 23.4, Cui et al. 2008) gives K / tau = gamma r / l0 * eta; eta (series tendon) is
#   calibrated on the ankle (soleus r 0.05 m, l0 0.044 m -> 26.6 /rad, eta = 3.5 / 26.6 = 0.13) and assumed 0.25 for the
#   shorter knee / hip tendons: vasti r 0.04, l0 0.09 -> 2.6; gluteus maximus r 0.06, l0 0.15 -> 2.3, iliopsoas 1.8;
#   gluteus medius r 0.05, l0 0.075 -> 3.9 (taken 2.5). Pfeifer et al. 2012 confirm knee stiffness is a continuous,
#   near-linear function of torque without co-contraction but give no slope in the text.
# Subtalar_x c = 3.0 (medium-low): Lee, Krebs, Hogan 2014 +1.1 N m/rad per % MVC (inversion-eversion), MVC 30-40 N m
#   (assumption) -> 2.8-3.7.
# k_base: Hip_y 35 (low end of the measured walking swing range), Knee_y 1.5 (walking swing 0.5-3.5), Ankle_y 22
#   (0.35 N m/rad/kg x 63.3 kg, low end of Lee & Hogan), Hip_x 34 (assumption = sagittal swing), Subtalar_x 8.8 (Ho 2016
#   relaxed). k_co_max (u = 1) defaults to max over states of the HMV-ImpRef upper end minus k_base. d (k_d / sqrt(k_p))
#   defaults to the geometric mean over states of b_centre / sqrt(k_centre).
VIC3_PARAMS = {
    "Hip_y": dict(k_base=35.0, c=2.0),
    "Knee_y": dict(k_base=1.5, c=2.5),
    "Ankle_y": dict(k_base=22.0, c=3.5),
    "Hip_x": dict(k_base=34.0, c=2.5),
    "Subtalar_x": dict(k_base=8.8, c=3.0),
}


def vic3_defaults(key: str, overrides: dict = None) -> dict:
    """Per-joint v3 parameters (k_base, c, k_co_max, d) with the HMV-ImpRef ranges; overrides by joint key."""
    ranges = dict(HUMAN_RANGES, **HUMAN_RANGES_FRONTAL)
    if key not in ranges or key not in VIC3_PARAMS:
        raise ValueError(f"vic_tc_v3 has no parameters for {key}")
    p = dict(VIC3_PARAMS[key], **((overrides or {}).get(key, {})))
    r = ranges[key]
    p.setdefault("k_co_max", max(hi for _, hi in r["k"]) - p["k_base"])
    if "d" not in p:
        ratios = [math.sqrt(blo * bhi) / math.sqrt(math.sqrt(lo * hi)) for (lo, hi), (blo, bhi) in zip(r["k"], r["b"])]
        p["d"] = math.exp(sum(math.log(x) for x in ratios) / len(ratios))
    p["k"], p["b"] = r["k"], r["b"]
    return p


def vic3_stiffness(k_base: Tensor, c: Tensor, tau_load: Tensor, k_co_max: Tensor, u: Tensor,
                   qd: Tensor = None, srs_omega: float = 0.0):
    """v3 stiffness terms before the corridor: (k_base, k_load, k_co), each [N, J]."""
    g = 1.0 if (srs_omega <= 0 or qd is None) else srs_omega / (srs_omega + qd.abs())
    k_load = c * tau_load * g
    k_co = u * k_co_max
    return k_base.expand_as(k_load), k_load, k_co


def vic3_guard(log_k: Tensor, log_lo: Tensor, log_hi: Tensor) -> Tensor:
    """Dead-zone guard in octaves: 0 inside [lo, hi], log2 distance outside, clamped at 4, mean over joints."""
    out = torch.relu(log_k - log_hi) + torch.relu(log_lo - log_k)
    return (out / math.log(2.0)).clamp(max=4.0).mean(-1)


def vic_cost(value: Tensor) -> Tensor:
    """Penalty reward: -value (value >= 0)."""
    return -value


def fill_missing_fields(cfg):
    """Configs pickled before a field existed (resolved_configs.pt of earlier runs) lack fields that use a
    default_factory (no class attribute to fall back on); give them the dataclass default. 2026-10-28."""
    import dataclasses
    if not dataclasses.is_dataclass(cfg):
        return cfg
    for f in dataclasses.fields(type(cfg)):
        if f.name not in vars(cfg) and not hasattr(type(cfg), f.name):
            if f.default_factory is not dataclasses.MISSING:
                setattr(cfg, f.name, f.default_factory())
            elif f.default is not dataclasses.MISSING:
                setattr(cfg, f.name, f.default)
    return cfg


@dataclass
class VicSteeringConfig(CadenceGaitPercentSteeringConfig):
    _target_: str = "protomotions.envs.control.vic_steering.VicSteering"
    vic_joints: tuple = ("L_Hip_y", "L_Knee_y", "L_Ankle_y", "R_Hip_y", "R_Knee_y", "R_Ankle_y",
                         "L_Hip_x", "R_Hip_x", "L_Subtalar_x", "R_Subtalar_x")
    delta_max: float = 1.1
    tau_res_scale: float = 0.33          # fraction of the joint strength cap
    tau_ff_lowpass_s: float = 0.03       # activation-like lowpass on tau_res (per physics substep)
    gain_smooth_s: float = 0.1
    band_widen: float = 1.5              # tolerance: human range widened by this factor each side
    frontal_band: float = 2.0            # tolerance for joints whose reference is the current gain
    stand_speed: float = 0.1
    stand_root_speed: float = 0.25
    # Schedule in control steps (one env.step = one control step for all envs).
    baseline_steps: int = 600
    alpha_steps: int = 9000
    beta_steps: int = 9000
    beta_decay: bool = True
    alpha_init: float = 0.0
    beta_init: float = 1.0
    schedule: bool = True
    fall_ratio_tol: float = 1.5
    fall_abs_tol: float = 0.0005
    log_every: int = 200
    # vic_TC_r2 cost inputs (rewards only exist when a recipe adds them). Human walking step peak of the vertical
    # GRF is 1.1-1.3 body weights; metabolic proxy P+/0.25 + |P-|/1.20 (muscle efficiencies, Margaria 1968) of
    # Hip_y, Knee_y, Ankle_y, in units of m*g*(1 m/s).
    body_mass: float = 63.3
    grf_ref_bw: float = 1.3
    grf_bodies: tuple = (("L_Ankle", "L_Toe"), ("R_Ankle", "R_Toe"))
    # k_d = exp(lb + damp_delta_gain * delta): 1 scales damping with stiffness (zeta grows as sqrt(k)); 0.5 keeps the
    # damping ratio zeta of the reference while the policy changes k (k_d ~ sqrt(k_p)), vic_TC_r4 (2026-10-23).
    damp_delta_gain: float = 1.0
    fresh_stance: bool = False           # contact from foot heights on the first step after a reset (r2 trains with it)
    # Candidate change (2026-10-26, HMV_v1 meeting follow-up Q2; defaults keep r1-r10 behaviour):
    # delta_band: delta_max per joint and leg state = half log-width of the human range + ln(delta_band_margin), so
    #   every state can reach its whole range (ankle stance needs 1.46 > 1.1); joints without a human value keep
    #   delta_max. The bound follows the smoothed state weights, so it changes as smoothly as the reference.
    # delta_lowpass_s: first-order lowpass of delta per control step (muscle activation ~30-50 ms); 0 = off.
    delta_band: bool = False
    frontal_human_ref: bool = False      # Hip_x / Subtalar_x follow HUMAN_RANGES_FRONTAL instead of the current gain
    delta_band_margin: float = 1.5
    delta_lowpass_s: float = 0.0
    # Activation-like asymmetric lowpass (2026-10-26, overrides delta_lowpass_s when both > 0): delta read as the
    # co-contraction level; rising (stiffening) follows tau_up, falling (relaxing) tau_down, like muscle activation
    # vs deactivation (~10-20 ms vs ~30-60 ms, Zajac 1989; Thelen 2003). Gains change once per control step.
    delta_tau_up_s: float = 0.0
    delta_tau_down_s: float = 0.0
    coact_unit_cost: bool = False        # vic_tc_v2: cocontraction cost on the level u = delta / h (not on delta)
    # vic_tc_v3 (2026-10-28, candidate; see module docstring). "v2" = everything above, unchanged.
    vic_law: str = "v2"
    v3_params: dict = field(default_factory=dict)   # per joint key overrides of VIC3_PARAMS (k_base, c, k_co_max, d)
    v3_hard_lo: float = 0.3              # hard corridor: [v3_hard_lo * lo_s, v3_hard_hi * hi_s] of the HMV-ImpRef range
    v3_hard_hi: float = 2.0
    v3_guard_margin: float = 1.25        # guard dead zone [lo / m, hi * m]
    v3_load_tau_up_s: float = 0.05       # activation-like filter of |tau| for the load term
    v3_load_tau_down_s: float = 0.15
    v3_load_source: str = "sim"          # "sim": simulator.human_active_torques; "estimate": own PD + ff estimate
    v3_srs_omega: float = 0.0            # rad/s; > 0 attenuates the load term during fast motion
    v3_u_map: str = "relu"               # "relu": u = relu(tanh(a)); "affine": u = (tanh(a) + 1) / 2
    v3_coact_cost: str = "torque"        # "torque": mean_j k_co / (c tau_max); "unit": mean_j u


class VicSteering(CadenceGaitPercentSteering):
    def __init__(self, config, env):
        fill_missing_fields(config)
        super().__init__(config, env)
        c = config
        sim = env.simulator
        names = list(env.robot_config.kinematic_info.dof_names)
        dev = self._tar_speed.device
        self._idx = torch.tensor([names.index(j) for j in c.vic_joints], device=dev)
        self._leg = torch.tensor([0 if j.startswith("L_") else 1 for j in c.vic_joints], device=dev)
        J = len(c.vic_joints)
        self.extra_action_dim = 2 * J
        info = env.robot_config.control.control_info
        k_old = torch.tensor([info[j].stiffness for j in c.vic_joints], device=dev, dtype=torch.float)
        b_old = torch.tensor([info[j].damping for j in c.vic_joints], device=dev, dtype=torch.float)
        tau_max = torch.tensor([info[j].effort_limit for j in c.vic_joints], device=dev, dtype=torch.float)
        self._k_old, self._b_old, self._tau_max = k_old, b_old, tau_max
        lk_h = torch.zeros(J, 3, device=dev)
        lb_h = torch.zeros(J, 3, device=dev)
        half = torch.zeros(J, 3, device=dev)       # log half-width of the tolerated band
        ranges = dict(HUMAN_RANGES, **(HUMAN_RANGES_FRONTAL if c.frontal_human_ref else {}))
        for i, j in enumerate(c.vic_joints):
            key = j[2:]
            if key in ranges:
                for s in range(3):
                    lo, hi = ranges[key]["k"][s]
                    blo, bhi = ranges[key]["b"][s]
                    lk_h[i, s] = 0.5 * (math.log(lo) + math.log(hi))
                    lb_h[i, s] = 0.5 * (math.log(blo) + math.log(bhi))
                    half[i, s] = 0.5 * (math.log(hi) - math.log(lo)) + math.log(c.band_widen)
            else:
                lk_h[i] = math.log(float(k_old[i]))
                lb_h[i] = math.log(float(b_old[i]))
                half[i] = math.log(c.frontal_band)
        self._lk_h, self._lb_h, self._half = lk_h, lb_h, half
        dmax = torch.full((J, 3), float(c.delta_max), device=dev)
        if c.delta_band:
            for i, j in enumerate(c.vic_joints):
                key = j[2:]
                if key in ranges:
                    for s in range(3):
                        lo, hi = ranges[key]["k"][s]
                        dmax[i, s] = 0.5 * (math.log(hi) - math.log(lo)) + math.log(c.delta_band_margin)
        self._dmax_h = dmax
        n = len(self._tar_speed)
        self._lk = torch.log(k_old).expand(n, J).clone()
        self._lb = torch.log(b_old).expand(n, J).clone()
        self._fresh = torch.ones(n, dtype=torch.bool, device=dev)
        self._dmax = dmax[:, 0].expand(n, J).clone()          # smoothed per-env bound (delta_band)
        self._delta = torch.zeros(n, J, device=dev)            # lowpassed delta (delta_lowpass_s)
        self.alpha = float(c.alpha_init)
        self.beta = float(c.beta_init)
        self._steps = 0
        self._fall_base = None
        self._fall_sum = 0.0
        self._fall_ema = None
        self._held = 0
        self._fall_acc = 0
        self._band = torch.zeros(n, device=dev)
        self._cocon = torch.zeros(n, device=dev)
        self._effort = torch.zeros(n, device=dev)
        self._mean_k = torch.zeros(3, J, device=dev)
        self._state_file = os.environ.get("HC_VIC_STATE_FILE")
        # HC_VIC_FIXED="alpha,beta" holds both values (evaluation of a trained checkpoint, smoke tests).
        fixed = os.environ.get("HC_VIC_FIXED")
        self._fixed = bool(fixed)
        if fixed:
            self.alpha, self.beta = (float(x) for x in fixed.split(","))
        # HC_VIC_FRESH_STANCE=1 (evaluation, 2026-10-23): on the first control step after a reset the cadence
        # contact flags are still False (both legs "swing", knee k ~ 2 N m/rad); take them from foot heights instead.
        self._fresh_stance = c.fresh_stance or os.environ.get("HC_VIC_FRESH_STANCE", "0") == "1"
        self._sag = torch.tensor([i for i, j in enumerate(c.vic_joints) if j[2:] in HUMAN_RANGES], device=dev)
        self._bw = c.body_mass * 9.81
        self._impact = torch.zeros(n, device=dev)
        self._metab = torch.zeros(n, device=dev)
        self._sim_ready = False
        # HC_VIC_GAIN_LOG=<path.npz> (evaluation, 2026-10-27, off by default): k_p, k_d, delta and leg state of the VIC
        # joints at every control step, written at exit (jitter analysis of the stiffness output).
        self._gain_log_path = os.environ.get("HC_VIC_GAIN_LOG")
        self._gain_log = []
        self._u_log = []                                       # vic_tc_v2: filtered co-activation level u [T, N, J]
        if self._gain_log_path:
            atexit.register(self._dump_gain_log)
        if c.vic_law == "v3":
            self._init_v3(n, dev)
        elif c.vic_law != "v2":
            raise ValueError(f"vic_law must be v2 or v3, got {c.vic_law}")

    def _init_v3(self, n, dev):
        c = self.config
        if c.v3_u_map not in ("relu", "affine") or c.v3_coact_cost not in ("torque", "unit") \
                or c.v3_load_source not in ("sim", "estimate"):
            raise ValueError("bad v3 option")
        ps = [vic3_defaults(j[2:], c.v3_params) for j in c.vic_joints]
        t = lambda v: torch.tensor(v, device=dev, dtype=torch.float)
        self._v3_kb = t([p["k_base"] for p in ps])
        self._v3_c = t([p["c"] for p in ps])
        self._v3_kco = t([p["k_co_max"] for p in ps])
        self._v3_d = t([p["d"] for p in ps])
        lg = lambda f: torch.log(t([[f(p, s) for s in range(3)] for p in ps]))        # [J, 3]
        m = c.v3_guard_margin
        self._v3_hard = torch.stack([lg(lambda p, s: c.v3_hard_lo * p["k"][s][0]), lg(lambda p, s: c.v3_hard_hi * p["k"][s][1]),
                                     lg(lambda p, s: c.v3_hard_lo * p["b"][s][0]), lg(lambda p, s: c.v3_hard_hi * p["b"][s][1]),
                                     lg(lambda p, s: p["k"][s][0] / m), lg(lambda p, s: p["k"][s][1] * m)], 0)   # [6, J, 3]
        J = len(c.vic_joints)
        self._v3_bounds = torch.zeros(6, n, J, device=dev)     # smoothed per env (gain_smooth_s glide on state changes)
        self._v3_tau = torch.zeros(n, J, device=dev)           # filtered |tau| of the load term
        self._v3_tau_est = torch.zeros(n, J, device=dev)
        self._v3_terms = torch.zeros(3, n, J, device=dev)      # k_base, k_load, k_co (last step)
        self._v3_log = []

    # ------------------------------------------------------------------ simulator buffers
    def _setup_sim(self):
        sim = self.env.simulator
        n = len(self._tar_speed)
        p, d = sim._common_p_gains, sim._common_d_gains
        if p.dim() == 1:
            sim._common_p_gains = p.unsqueeze(0).expand(n, -1).clone()
            sim._common_d_gains = d.unsqueeze(0).expand(n, -1).clone()
        ndof = sim._common_p_gains.shape[1]
        z = torch.zeros(n, ndof, device=p.device)
        sim.vic_ff_prior, sim.vic_ff_res_target, sim.vic_ff_res = z.clone(), z.clone(), z.clone()
        dt = 1.0 / sim.config.sim.fps
        sim.vic_ff_coeff = torch.tensor(1.0 - math.exp(-dt / max(self.config.tau_ff_lowpass_s, 1e-6)),
                                        device=p.device)
        robot = sim._robot
        self._to_sim = sim.data_conversion.dof_convert_to_sim
        self._host_k = torch.empty((n, ndof), dtype=torch.float, pin_memory=True)
        self._host_d = torch.empty((n, ndof), dtype=torch.float, pin_memory=True)
        self._host_idx = robot._ALL_INDICES.cpu()
        self._sim_ready = True

    def _write_gains(self):
        sim = self.env.simulator
        robot = sim._robot
        k_sim = sim._common_p_gains[:, self._to_sim]
        d_sim = sim._common_d_gains[:, self._to_sim]
        robot._data.joint_stiffness.copy_(k_sim)
        robot._data.joint_damping.copy_(d_sim)
        self._host_k.copy_(k_sim)
        self._host_d.copy_(d_sim)
        robot.root_physx_view.set_dof_stiffnesses(self._host_k, indices=self._host_idx)
        robot.root_physx_view.set_dof_dampings(self._host_d, indices=self._host_idx)

    def _dump_gain_log(self):
        if not self._gain_log:
            return
        import numpy as np
        a = torch.stack(self._gain_log, 0).float().numpy()          # [T, 4, N, J]
        extra = {"u": torch.stack(self._u_log[:len(a)], 0).float().numpy()} if self._u_log else {}
        if getattr(self, "_v3_log", None):
            v = torch.stack(self._v3_log[:len(a)], 0).float().numpy()             # [T, 4, N, J]
            extra.update(k_base=v[:, 0], k_load=v[:, 1], k_co=v[:, 2], tau_load=v[:, 3])
        np.savez_compressed(self._gain_log_path, kp=a[:, 0], kd=a[:, 1], delta=a[:, 2], state=a[:, 3].astype("int8"),
                            joints=np.array(self.config.vic_joints), dt=float(self.env.dt), **extra)

    # ------------------------------------------------------------------ schedule
    def _advance_schedule(self):
        c = self.config
        self._steps += 1
        # Falls per env-step, counted when an env is reset after a fall termination (same quantity as the
        # training log's termination/fallen_mean). Reading terminate_buf before physics gave ~20x that value.
        fell = self._fall_acc / len(self._tar_speed)
        self._fall_acc = 0
        if self._steps <= c.baseline_steps:
            self._fall_sum += fell
            if self._steps == c.baseline_steps:
                self._fall_base = self._fall_sum / c.baseline_steps
                self._fall_ema = self._fall_base
            return
        if self._fall_ema is None:
            self._fall_ema = fell
        self._fall_ema += (fell - self._fall_ema) / 300.0
        if not c.schedule or self._fixed or self._fall_base is None:
            return
        if self._fall_ema > c.fall_ratio_tol * self._fall_base + c.fall_abs_tol:
            self._held += 1
            return
        if self.alpha < 1.0:
            self.alpha = min(1.0, self.alpha + 1.0 / c.alpha_steps)
        elif c.beta_decay and self.beta > 0.0:
            self.beta = max(0.0, self.beta - 1.0 / c.beta_steps)

    def _log(self):
        c = self.config
        if self._steps % c.log_every:
            return
        mk = self._mean_k.exp().cpu().numpy().round(2).tolist()
        rec = dict(step=self._steps, alpha=round(self.alpha, 4), beta=round(self.beta, 4),
                   fall_rate=None if self._fall_ema is None else round(self._fall_ema, 5),
                   fall_base=None if self._fall_base is None else round(self._fall_base, 5),
                   held_steps=self._held, band_cost=round(self._band.mean().item(), 4),
                   cocontraction=round(self._cocon.mean().item(), 4), effort=round(self._effort.mean().item(), 4),
                   impact=round(self._impact.mean().item(), 4), metab=round(self._metab.mean().item(), 4),
                   mean_kp_swing_stance_stand=mk, joints=list(c.vic_joints))
        if c.vic_law == "v3":
            t = self._v3_terms.mean(1).cpu().numpy().round(2)
            rec.update(v3_mean_k_base=t[0].tolist(), v3_mean_k_load=t[1].tolist(), v3_mean_k_co=t[2].tolist())
        print("[vic] " + json.dumps(rec), flush=True)
        if self._state_file:
            try:
                with open(self._state_file, "w") as f:
                    json.dump(rec, f)
            except OSError:
                pass

    # ------------------------------------------------------------------ main hook
    def reset(self, env_ids):
        if len(env_ids) and hasattr(self.env, "terminate_buf"):
            # Called before the env clears terminate_buf; skip envs that never ran (initial reset).
            fell = self.env.terminate_buf[env_ids] & (self.env.progress_buf[env_ids] > 2)
            self._fall_acc = getattr(self, "_fall_acc", 0) + int(fell.sum().item())
        super().reset(env_ids)
        if len(env_ids):
            self._fresh[env_ids] = True

    def pre_physics_extra_action(self, extra: Tensor, q_ref: Tensor):
        """Called by the env each control step before physics, with the raw extra outputs and the
        processed PD targets (common dof order)."""
        c = self.config
        if not self._sim_ready:
            self._setup_sim()
        self._advance_schedule()
        sim = self.env.simulator
        J = len(c.vic_joints)
        unit = torch.tanh(extra[:, :J])
        tau_res = c.tau_res_scale * self._tau_max * torch.tanh(extra[:, J:2 * J])
        # Leg state: 0 swing, 1 stance, 2 standing (both feet down, slow command and body).
        contact = self._stance
        if self._fresh_stance and self._fresh.any():
            height = sim.get_robot_state().rigid_body_pos[..., 2]
            measured = torch.stack([(height[:, idx] < self._stance_heights).any(-1) for idx in self._feet], dim=-1)
            contact = torch.where(self._fresh[:, None], measured, contact)
        stance = contact[:, self._leg]                                        # [N, J]
        root_v = sim.get_robot_state().rigid_body_vel[:, 0, :2].norm(dim=-1)
        standing = contact.all(-1) & (self._tar_speed < c.stand_speed) & (root_v < c.stand_root_speed)
        state = torch.where(stance, torch.ones_like(stance, dtype=torch.long),
                            torch.zeros_like(stance, dtype=torch.long))
        state = torch.where(standing[:, None], torch.full_like(state, 2), state)
        jj = torch.arange(J, device=state.device).expand_as(state)
        lk_h = self._lk_h[jj, state]
        lb_h = self._lb_h[jj, state]
        a = self.alpha
        lk_t = (1 - a) * torch.log(self._k_old) + a * lk_h
        lb_t = (1 - a) * torch.log(self._b_old) + a * lb_h
        s = 1.0 - math.exp(-self.env.dt / c.gain_smooth_s)
        self._lk = torch.where(self._fresh[:, None], lk_t, self._lk + s * (lk_t - self._lk))
        self._lb = torch.where(self._fresh[:, None], lb_t, self._lb + s * (lb_t - self._lb))
        fresh = self._fresh.clone()
        self._fresh[:] = False
        if c.vic_law == "v3":
            return self._pre_physics_v3(extra, q_ref, unit, tau_res, state, jj, fresh, s)
        if c.delta_band:
            dm_t = self._dmax_h[jj, state]
            self._dmax = torch.where(fresh[:, None], dm_t, self._dmax + s * (dm_t - self._dmax))
            delta = self._dmax * unit
        else:
            delta = c.delta_max * unit
        if c.delta_tau_up_s > 0 and c.delta_tau_down_s > 0:
            up = 1.0 - math.exp(-self.env.dt / c.delta_tau_up_s)
            down = 1.0 - math.exp(-self.env.dt / c.delta_tau_down_s)
            step = torch.where(delta > self._delta, up, down) * (delta - self._delta)
            self._delta = torch.where(fresh[:, None], delta, self._delta + step)
            delta = self._delta
        elif c.delta_lowpass_s > 0:
            sd = 1.0 - math.exp(-self.env.dt / c.delta_lowpass_s)
            self._delta = torch.where(fresh[:, None], delta, self._delta + sd * (delta - self._delta))
            delta = self._delta
        kp = torch.exp(self._lk + delta)
        kd = torch.exp(self._lb + c.damp_delta_gain * delta)
        if self._gain_log_path:
            self._gain_log.append(torch.stack([kp, kd, delta, state.float()], 0).half().cpu())
            if c.coact_unit_cost:
                self._u_log.append((delta / self._dmax.clamp_min(1e-6)).half().cpu())
            if len(self._gain_log) % 250 == 0:      # the simulator may exit without running atexit hooks
                self._dump_gain_log()
        sim._common_p_gains[:, self._idx] = kp
        sim._common_d_gains[:, self._idx] = kd
        self._write_gains()
        dof = sim._get_simulator_dof_state().convert_to_common(sim.data_conversion)
        q, qd = dof.dof_pos[:, self._idx], dof.dof_vel[:, self._idx]
        err = q_ref[:, self._idx] - q
        prior = (self._k_old - kp) * err - (self._b_old - kd) * qd
        sim.vic_ff_prior[:, self._idx] = self.beta * prior
        sim.vic_ff_res_target[:, self._idx] = tau_res
        reset_now = self.env.progress_buf <= 1
        if reset_now.any():
            sim.vic_ff_res[reset_now] = 0.0
        # Reward inputs.
        dev_ = (self._lk + delta) - lk_t
        self._band = (torch.relu(dev_.abs() - self._half[jj, state]) / math.log(2.0)).clamp(max=4.0).mean(-1)
        self._cocon = (torch.relu(delta / self._dmax.clamp_min(1e-6)) if c.coact_unit_cost else torch.relu(delta)).mean(-1)
        tau = kp * err - kd * qd + self.beta * prior + sim.vic_ff_res[:, self._idx]
        self._effort = ((tau / self._tau_max) ** 2).clamp(max=4.0).mean(-1)
        p = (tau * qd)[:, self._sag]
        self._metab = (p.clamp(min=0).sum(-1) / 0.25 + (-p).clamp(min=0).sum(-1) / 1.20) / self._bw
        for st in range(3):
            m = state == st
            cnt = m.float().sum(0).clamp_min(1)
            self._mean_k[st] = torch.where(m.any(0), ((self._lk + delta) * m).sum(0) / cnt, self._mean_k[st])
        self._log()

    def coactivation_level(self):
        """Filtered co-activation level per VIC joint for the plant's joint reflex (0 = relaxed, 1 = maximal)."""
        if getattr(self.config, "vic_law", "v2") == "v3":
            return self._delta.clamp(0.0, 1.0)                 # v3 keeps u in [0, 1] in _delta
        return (self._delta / self._dmax.clamp_min(1e-6)).clamp(-1.0, 1.0)

    def _pre_physics_v3(self, extra, q_ref, unit, tau_res, state, jj, fresh, s):
        """vic_tc_v3 gains (module docstring); same feed-forward, reward inputs and logs as v2 otherwise."""
        c = self.config
        sim = self.env.simulator
        dt = self.env.dt
        # Co-contraction level u in [0, 1], activation-like filter as v2 (delta_tau_up_s / delta_tau_down_s).
        u = torch.relu(unit) if c.v3_u_map == "relu" else 0.5 * (unit + 1.0)
        if c.delta_tau_up_s > 0 and c.delta_tau_down_s > 0:
            up = 1.0 - math.exp(-dt / c.delta_tau_up_s)
            down = 1.0 - math.exp(-dt / c.delta_tau_down_s)
            step = torch.where(u > self._delta, up, down) * (u - self._delta)
            self._delta = torch.where(fresh[:, None], u, self._delta + step)
            u = self._delta
        # Load term input: active torque of the previous control step (no algebraic loop).
        src = getattr(sim, "human_active_torques", None) if c.v3_load_source == "sim" else None
        if src is not None and src.shape[0] == len(fresh):
            tau_prev = src[:, self._idx].detach().abs()
        else:
            tau_prev = self._v3_tau_est.abs()
        lup = 1.0 - math.exp(-dt / c.v3_load_tau_up_s)
        ldn = 1.0 - math.exp(-dt / c.v3_load_tau_down_s)
        self._v3_tau = self._v3_tau + torch.where(tau_prev > self._v3_tau, lup, ldn) * (tau_prev - self._v3_tau)
        self._v3_tau = torch.where(fresh[:, None], torch.zeros_like(self._v3_tau), self._v3_tau)   # stale after reset
        dof = sim._get_simulator_dof_state().convert_to_common(sim.data_conversion)
        q, qd = dof.dof_pos[:, self._idx], dof.dof_vel[:, self._idx]
        kb, kl, kc = vic3_stiffness(self._v3_kb, self._v3_c, self._v3_tau, self._v3_kco, u, qd, c.v3_srs_omega)
        # Corridor and guard bounds of the leg state, gliding over gain_smooth_s like the v2 reference.
        tgt = self._v3_hard[:, jj, state]                                       # [6, N, J]
        self._v3_bounds = torch.where(fresh[None, :, None], tgt, self._v3_bounds + s * (tgt - self._v3_bounds))
        lo_k, hi_k, lo_b, hi_b, g_lo, g_hi = self._v3_bounds
        lk3 = torch.log(kb + kl + kc).clamp(lo_k, hi_k)
        lb3 = (torch.log(self._v3_d) + 0.5 * lk3).clamp(lo_b, hi_b)
        a = self.alpha
        lk = (1 - a) * torch.log(self._k_old) + a * lk3
        lb = (1 - a) * torch.log(self._b_old) + a * lb3
        kp, kd = torch.exp(lk), torch.exp(lb)
        self._lk, self._lb = lk, lb
        self._v3_terms = torch.stack([kb, kl, kc], 0)
        delta = lk - torch.log(self._v3_kb)                  # gain log "delta": log(k_p / k_base)
        if self._gain_log_path:
            self._gain_log.append(torch.stack([kp, kd, delta, state.float()], 0).half().cpu())
            self._u_log.append(u.half().cpu())
            self._v3_log.append(torch.stack([kb, kl, kc, self._v3_tau], 0).half().cpu())
            if len(self._gain_log) % 250 == 0:
                self._dump_gain_log()
        sim._common_p_gains[:, self._idx] = kp
        sim._common_d_gains[:, self._idx] = kd
        self._write_gains()
        err = q_ref[:, self._idx] - q
        prior = (self._k_old - kp) * err - (self._b_old - kd) * qd
        sim.vic_ff_prior[:, self._idx] = self.beta * prior
        sim.vic_ff_res_target[:, self._idx] = tau_res
        reset_now = self.env.progress_buf <= 1
        if reset_now.any():
            sim.vic_ff_res[reset_now] = 0.0
        # Reward inputs: dead-zone guard (never toward the centre), co-contraction cost, effort, metabolic proxy.
        self._band = vic3_guard(lk, g_lo, g_hi)
        if c.v3_coact_cost == "torque":
            self._cocon = (kc / (self._v3_c * self._tau_max)).mean(-1)
        else:
            self._cocon = u.mean(-1)
        tau = kp * err - kd * qd + self.beta * prior + sim.vic_ff_res[:, self._idx]
        self._v3_tau_est = torch.maximum(torch.minimum(tau, self._tau_max), -self._tau_max).detach()
        self._effort = ((tau / self._tau_max) ** 2).clamp(max=4.0).mean(-1)
        p = (tau * qd)[:, self._sag]
        self._metab = (p.clamp(min=0).sum(-1) / 0.25 + (-p).clamp(min=0).sum(-1) / 1.20) / self._bw
        for st in range(3):
            m = state == st
            cnt = m.float().sum(0).clamp_min(1)
            self._mean_k[st] = torch.where(m.any(0), (lk * m).sum(0) / cnt, self._mean_k[st])
        self._log()

    def _update_impact(self):
        """Per leg, peak vertical contact force over the physics substeps of the last control step (body weights);
        cost = mean over legs of max(0, peak - grf_ref_bw), clamped at 3."""
        sm = getattr(self.env.simulator, "_contact_sensor_map", None)
        if not sm:
            return
        legs = [sum(sm[b].data.net_forces_w_history[:, :, 0, 2] for b in bodies).amax(-1) for bodies in self.config.grf_bodies]
        peak = torch.stack(legs, -1) / self._bw
        self._impact = torch.relu(peak - self.config.grf_ref_bw).clamp(max=3.0).mean(-1)

    def populate_context(self, ctx):
        super().populate_context(ctx)
        self._update_impact()
        ctx.steering.vic_impact = self._impact
        ctx.steering.vic_metab = self._metab
        ctx.steering.vic_band_cost = self._band
        ctx.steering.vic_cocontraction = self._cocon
        ctx.steering.vic_effort = self._effort
