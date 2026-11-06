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
Parameters and sources: VIC3_PARAMS below. v3_load_gate = "stance" (candidate, 2026-10-29) feeds the load filter only in
stance / standing, so swing k = k_base + k_co (vic3_TC_s5_r1 swing knee load term pinned k to the corridor ceiling).

vic_tc_v4 (2026-10-30, user-approved version: impedance prior; vic_law = "v4", default "v2" unchanged). The total joint
stiffness, plant passive tissue + VIC, follows the human impedance reference through a reward, like AMP for motion:

    k_tot,j = k_pass,j(q) + k_act,j,      ln k_tot,j = ln k_c,j(phi) + H_j(phi) * a_j,   H = h(phi) + ln(v4_band_margin)
    k_act,j = max(k_tot,j - k_pass,j, v4_k_floor_frac * k_c,j)       (PD stiffness written to the simulator)
    ln b_tot,j = ln b_c,j(phi) + alpha_j (ln k_tot,j - ln k_c,j(phi)), clamped to [b_lo / 2, 2 b_hi](phi)   (D1)
    r_imp = exp(-mean_j (max(0, |ln k_eff,j - ln k_c,j(phi)| - h_j(phi)) / v4_imp_scale)^2),  k_eff = k_act + k_pass

k_pass: diagonal of the plant's passive tissue tangent stiffness (HumanJointModel exponential passive moments + linear
springs, human_model_v3 profile; the same forces the plant already applies), so the VIC no longer double-counts it
(v1-v3 put the whole human value into the PD gain on top of the plant's passive moments). b_pass = plant damping (0 for
the leg joints). The joint spinal reflex (human_model_v3.2) is threshold-gated and silent in steady walking, so it is
not added to k_eff; HMV-ImpRef values come from brief perturbations and may contain short-latency reflex, which the VIC
then covers (noted in share/training_configs/vic_TC_server; the impedance probe measures plant + reflex + VIC).
phi is continuous, no state switch: per leg the load fraction lambda = vertical GRF / body weight (lowpass
v4_load_tau_s; contact flags if the simulator has no contact sensors) gives w_stance = clip((lambda - lo) / (hi - lo));
w_stand = w_stance,L * w_stance,R * slow(command) * slow(body); weights (swing, stance, stand) =
((1 - w_stand)(1 - w_stance), (1 - w_stand) w_stance, w_stand) blend the HMV-ImpRef log centres and half-widths. Ankle
late stance (stance progress t / T_prev > v4_late_p) blends the stance range toward Shorter & Rouse 2018 late stance
(VIC4_ANKLE_LATE_K). The action a (tanh, activation-like filter delta_tau_up_s / delta_tau_down_s as v2) moves k inside
twice the human range (hard limit), the reward pulls only outside the range (dead zone, never toward the centre).
Co-contraction for J2M: tau_co = max(0, k_act - c |tau|) / c (stiffness beyond the load-dependent short-range stiffness
of the joint's own torque, c = VIC3_PARAMS), u = tau_co / tau_max in [0, 1]; the cocontraction cost is mean_j u.
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
VIC_SWAY = steering_path("vic_sway")
VIC_PULSE_ERR = steering_path("vic_pulse_err")
VIC_EFFERENCE = steering_path("vic_efference")
VIC_IMP_PRIOR = steering_path("vic_imp_prior")
VIC_MOMENT_PRIOR = steering_path("vic_moment_prior")   # 2026-11-05 candidate: human joint-moment profile reward
VIC_MOMENT_FEAT = steering_path("vic_moment_feat")     # 2026-11-06 candidate: leg joint moments for the AMP discriminator

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


# vic_tc_v4 damping exponent alpha_j in b ~ k^alpha (D1, TO BE DONE 2026-10-29): ankle b is nearly independent of k in
# people (HMV-ImpRef 0.6-1.5 N m s/rad over k 22-406; assumption-level fit), knee / hip keep the v1-v3 shape b ~ sqrt(k).
VIC4_ALPHA = {"Hip_y": 0.5, "Knee_y": 0.5, "Ankle_y": 0.15, "Hip_x": 0.5, "Subtalar_x": 0.5}
# Ankle late stance stiffness, Shorter & Rouse 2018 (HMV-ImpRef csv source [5], 63.3 kg): design data for the phase
# shape inside stance; the late-stance window (progress > v4_late_p) is an assumption.
VIC4_ANKLE_LATE_K = (133.0, 234.0)


def vic4_leg_weights(load: Tensor, slow: Tensor, lo: float, hi: float) -> Tensor:
    """Continuous leg-state weights [N, 2, 3] (swing, stance, stand) from per-leg load fractions [N, 2] and a standing
    slowness factor [N] in [0, 1]."""
    w_st = ((load - lo) / max(hi - lo, 1e-6)).clamp(0.0, 1.0)
    g = (w_st[:, 0] * w_st[:, 1] * slow)[:, None]
    return torch.stack([(1 - g) * (1 - w_st), (1 - g) * w_st, g.expand_as(w_st)], -1)


def vic4_blend(w: Tensor, table: Tensor) -> Tensor:
    """Blend a per-joint, per-state table [J, 3] (log values) with weights [N, J, 3] -> [N, J]."""
    return (w * table).sum(-1)


def vic4_passive_diag(model, q_all: Tensor, idx: Tensor):
    """Diagonal passive tissue stiffness and damping of the plant (HumanJointModel.passive_impedance, diagonal only,
    without building the full matrix) at the VIC joints: ([N, J], [J])."""
    z = q_all @ model.a.T + model.offset
    thr = math.log(model.exp_cap)
    wgt = model.energy_scale * torch.exp(z.clamp(max=thr)) * (z < thr)
    k = wgt @ (model.a[:, idx] ** 2)
    disp = q_all[:, idx] - model.rest[idx]
    kl = torch.where(disp < 0, model.k_negative[idx], model.k_positive[idx])
    k = k + kl * ((kl * disp).abs() < model.spring_cap)
    return k, model.damping[idx]


def vic4_passive_torque(model, q_all: Tensor, idx: Tensor) -> Tensor:
    """Plant passive tissue torque at the VIC joints [N, J] (exponential terms of HumanJointModel, -dE/dq, plus the linear
    springs, capped), the same forces the plant applies; damping omitted (0 for the leg joints)."""
    z = q_all @ model.a.T + model.offset
    thr = math.log(model.exp_cap)
    wgt = model.energy_scale * torch.exp(z.clamp(max=thr))
    tau = -(wgt @ model.a[:, idx])
    disp = q_all[:, idx] - model.rest[idx]
    kl = torch.where(disp < 0, model.k_negative[idx], model.k_positive[idx])
    return tau - (kl * disp).clamp(-model.spring_cap, model.spring_cap)


_MOMENT_TABLES = {}


def vic_moment_prior(tau_nkg: Tensor, tar_speed: Tensor, gait_pct: Tensor, gait_pct_valid: Tensor, table_file: str,
                     signs: tuple, sd_floor: float, kernel_scale: float) -> Tensor:
    """Human sagittal joint-moment profile reward (2026-11-05 candidate). tau_nkg [N, 2, 3]: per leg (L, R) hip / knee /
    ankle total joint torque in N m/kg in plant signs; the table (speeds, mean / sd [S, 101, 3] in its own signs) is mapped
    by signs. Per leg r = exp(-0.5 mean_j z_j^2), z = (tau - mean) / (kernel_scale * max(sd, sd_floor)) at the leg's gait
    percent and the nearest table speed; r = 1 where the gait percent is not valid; mean over legs."""
    key = (table_file, str(tau_nkg.device))
    if key not in _MOMENT_TABLES:
        t = torch.load(table_file, map_location=tau_nkg.device, weights_only=False)
        sg = torch.tensor(signs, device=tau_nkg.device, dtype=torch.float)
        _MOMENT_TABLES[key] = (t["speeds"].to(tau_nkg.device).float(), t["mean"].to(tau_nkg.device).float() * sg,
                               t["sd"].to(tau_nkg.device).float())
    speeds, mean, sd = _MOMENT_TABLES[key]
    s = torch.bucketize(tar_speed.float(), (speeds[1:] + speeds[:-1]) / 2).clamp(0, len(speeds) - 1)
    out = []
    for leg in range(2):
        p = (gait_pct[:, leg].clamp(0, 1) * 100).round().long()
        z = (tau_nkg[:, leg] - mean[s, p]) / (kernel_scale * sd[s, p].clamp_min(sd_floor))
        r = torch.exp(-0.5 * (z ** 2).mean(-1))
        out.append(torch.where(gait_pct_valid[:, leg], r, torch.ones_like(r)))
    return (out[0] + out[1]) / 2


def vic4_damping(lk: Tensor, lk_c: Tensor, lb_c: Tensor, alpha: Tensor, lb_lo: Tensor, lb_hi: Tensor) -> Tensor:
    """D1 damping law in log space: ln b = ln b_c + alpha (ln k - ln k_c), clamped to [ln b_lo - ln 2, ln b_hi + ln 2]."""
    return (lb_c + alpha * (lk - lk_c)).clamp(lb_lo - math.log(2.0), lb_hi + math.log(2.0))


def vic4_coactivation(k_act: Tensor, c: Tensor, tau_abs: Tensor, tau_max: Tensor):
    """Co-contraction torque tau_co = max(0, k_act - c |tau|) / c and level u = tau_co / tau_max in [0, 1]."""
    tau_co = torch.relu(k_act - c * tau_abs) / c
    return tau_co, (tau_co / tau_max).clamp(0.0, 1.0)


def vic4_dead_zone(lk: Tensor, lk_c: Tensor, h: Tensor) -> Tensor:
    """Log distance outside the human range [N, J] (0 inside, never toward the centre)."""
    return torch.relu((lk - lk_c).abs() - h)


def vic_gain(value: Tensor) -> Tensor:
    """Positive reward: value (in [0, 1])."""
    return value


def vic_cost(value: Tensor) -> Tensor:
    """Penalty reward: -value (value >= 0)."""
    return -value


def compute_vic_efference_obs(vic_efference: Tensor) -> Tensor:
    """Efference copy of the VIC outputs (u command and tau_ff target / tau_max, last vic_efference_steps steps)."""
    return vic_efference


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
    # 2026-11-03 candidate (vic_TC day run B): joints in beta_hold_joints keep beta_hold instead of the global beta (e.g.
    # the sagittal knees at 0.2: beta <= 0.1 collapsed the 1.4 m/s knee, beta 0.2 kept it; hip / ankle go to beta 0).
    # Default () = every joint uses the global beta (earlier runs unchanged).
    beta_hold_joints: tuple = ()
    beta_hold: float = 0.0
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
    # Load term gate (2026-10-29, candidate): "stance" feeds |tau| to the load filter only while the leg is in stance or
    # standing (swing input 0, so the term rises with the 50 ms filter after foot contact and decays over 150 ms after
    # toe-off). vic3_TC_s5_r1: swing torques (leg swinging, mostly the PD's own k * err) gave the swing knee 13.6 N m/rad
    # of load term vs the human 0.5-3.5, pinning k to the corridor ceiling 98 % of steps. "none" = r1 behaviour.
    v3_load_gate: str = "none"
    # Reasons for co-contraction (2026-10-29, candidates, off by default; user: u should rise where a person co-contracts).
    # Signal-dependent motor noise: torque noise SD = v3_motor_noise * (|tau| + tau_co) per VIC joint, tau_co = k_co / c
    # (agonist + antagonist torque that gives k_co, so co-contraction itself adds noise as in muscle), coloured by a
    # first-order filter (v3_motor_noise_tau_s, muscle twitch ~ 40 ms), added to the feed-forward target each control step.
    # Force variability grows with force (Jones, Hamilton, Wolpert 2002); stiffness lowers the effect of the noise on the
    # joint (Selen, Franklin, Wolpert 2009), with the sensory delay that is the reason to pre-stiffen.
    v3_motor_noise: float = 0.0
    v3_motor_noise_tau_s: float = 0.04
    # Quiet-standing sway cost: in the standing state (both feet down, slow command and body) relu(|v_root,xy| - v_h) / v_h,
    # clamped at 4; v_h = human quiet-stance centre-of-pressure mean speed scale (~10 mm/s, Prieto et al. 1996), dead zone
    # below it (no pull to zero sway). Written to ctx.steering.vic_sway (zero unless sway_cost).
    sway_cost: bool = False
    sway_v_h: float = 0.01
    v3_u_map: str = "relu"               # "relu": u = relu(tanh(a)); "affine": u = (tanh(a) + 1) / 2;
    #                                      "softplus": u = softplus(8 tanh(a)) / 8 (no flat zero region: u = 0.087 at a = 0)
    v3_coact_cost: str = "torque"        # "torque": mean_j k_co / (c tau_max); "unit": mean_j u;
    #   "activation" (2026-10-29): extra effort of co-contraction, mean_j [((|tau| + tau_co) / tau_max)^2 - (tau / tau_max)^2],
    #   i.e. the same quadratic activation price as vic_effort (weight it like vic_effort so u is not dearer than tau_ff)
    # Unpredictable joint torque pulses (2026-10-29, candidate, off): per VIC joint a pulse starts with rate
    # v3_perturb_rate_hz, sign random, size v3_perturb_amp * tau_max * U(0.5, 1), lasting v3_perturb_dur_s, only while the
    # joint's leg is standing (state 2) or in the first v3_perturb_loading_s of stance. Shorter than the afferent delay +
    # one control step, so q_ref cannot cancel it in time; stiffness does (unstable / unpredictable loads raise
    # co-contraction in people: Burdet et al. 2001, Franklin et al. 2003). Not the BS push (small, joint level).
    v3_perturb_rate_hz: float = 0.0
    v3_perturb_amp: float = 0.08
    v3_perturb_dur_s: float = 0.06
    v3_perturb_loading_s: float = 0.15
    # Evaluation diagnostic (2026-10-29, default 0): u <- clamp(u + v3_u_add, 0, 1) after the filter, to test whether a
    # higher co-contraction helps the trained policy at all.
    v3_u_add: float = 0.0
    # Kinematic error under a pulse (2026-10-29, candidate, off): while a perturbation pulse is live on a joint,
    # cost mean_j ((qdot_j - qdot_j at pulse onset) / 1 rad/s)^2, clamped at 4 (ctx.steering.vic_pulse_err). People raise
    # co-contraction in proportion to the kinematic error an unpredictable load causes (Franklin et al. 2008); without
    # it, the early-stance pulses only cost through falls and the gait prior.
    pulse_err_cost: bool = False
    # Efference copy of the VIC outputs (2026-10-29, candidate, 0 = off): keep the last N steps of the u command and of
    # tau_ff target / tau_max (2 x J per step) in ctx.steering.vic_efference for an actor observation.
    vic_efference_steps: int = 0
    # vic_tc_v4 (2026-10-30, user-approved version; module docstring). Used only with vic_law = "v4".
    v4_band_margin: float = 2.0          # hard limit: human range widened by this factor on each side
    v4_load_lo: float = 0.05             # load fraction (body weights) below which a leg is swing
    v4_load_hi: float = 0.20             # ... above which it is fully stance
    v4_load_tau_s: float = 0.03          # lowpass of the load fraction
    v4_late_anchor: bool = True          # ankle late-stance reference (VIC4_ANKLE_LATE_K)
    v4_late_p: tuple = (0.5, 0.65)       # stance progress ramp of the late-stance blend (assumption)
    v4_stance_T0_s: float = 0.65         # initial previous-stance duration (assumption, ~1.2 m/s walking)
    v4_k_floor_frac: float = 0.02        # PD stiffness floor, fraction of the reference centre
    v4_imp_scale: float = 0.5            # r_imp scale in ln units outside the range (0.5 -> factor 1.65)
    v4_passive: bool = True              # subtract the plant passive stiffness / damping from the PD gains
    v4_alpha: dict = field(default_factory=dict)   # per joint key overrides of VIC4_ALPHA
    # Scratch strategy A stage A1 (2026-10-30, candidate, default off): ignore the policy's stiffness outputs (a = 0), so
    # k_tot follows the human reference centre k_c(phi): a walking teacher on human impedance. Stage A2 sets it False.
    v4_freeze_a: bool = False
    # Beta prior per physics substep (2026-11-01, candidate, default off): the steering writes the gain gaps
    # beta (k_old - k_p), beta (b_old - k_d) and dynamics._vic_prior_substep applies them with each substep's error, so
    # beta = 1 equals the fixed PD instead of a torque held for the control step (v4 law only).
    v4_prior_substep: bool = False
    # Training-time gait-curve correlation (2026-11-02, logging only, empty = off): Fukuchi table (knee / ankle mean by
    # speed and gait %, the HMV-Walk knee r reference). Envs commanded within 0.1 m/s of each gait_r_speeds value at the
    # natural cadence fill a stride-averaged curve per speed; every gait_r_window control steps the Pearson r against the
    # table is logged as env/vic_gait/{knee,ankle}_r_vXX and the accumulators restart.
    # r_imp without the swing knee (2026-11-02, user-approved reward change, default off): the plant's passive knee
    # stiffness along human swing angles (6-24 N m/rad, Silder 2007 exponentials) is above the human swing reference
    # (0.5-3.5), so the knee term was a penalty no action could remove (W3 already leaves the swing knee out). The knee
    # terms are weighted by 1 - swing weight of their leg in the joint mean.
    v4_imp_skip_swing_knee: bool = False
    # Speed-dependent stance reference centre (2026-11-04, user-approved "속도별 강성 기준", default off = {}): in stance,
    # ln k_c += w_stance * gamma_j * ln(v_s / v4_kc_speed_ref), v_s = commanded speed lowpassed (v4_kc_speed_tau_s) and
    # clipped to v4_kc_speed_clip. gamma by joint suffix (e.g. {"Hip_y": 0.64, "Knee_y": 0.87, "Ankle_y": 0.52}): power-law
    # exponent of stance quasi-stiffness vs speed, WBDS (Fukuchi 2018) sagittal moments vs angles 0.6-1.6 m/s (hip angle from
    # Lencioni 2019); only the ratio is used, quasi-stiffness is not impedance. Range half width h and swing / standing
    # centres unchanged; the hard limit (v4_band_margin) moves with the centre.
    v4_kc_speed_gamma: dict = field(default_factory=dict)
    v4_kc_speed_ref: float = 1.0
    v4_kc_speed_clip: tuple = (0.6, 1.6)
    v4_kc_speed_tau_s: float = 0.3
    v4_kc_speed_damp: bool = False       # also move the damping centre by alpha_j x the stiffness shift (2026-11-04)
    # 2026-11-05 candidate option (user: "로컬 1안 ㄱㄱ"; no version bump until approved, default off): first-order lowpass
    # on the PD targets q_ref of the VIC (leg) joints at this cutoff [Hz], applied per control step before the simulator
    # step. q_ref then carries only slow intent; fast deviations are answered by k (and tau_ff), so the stiffness gets a
    # learning signal of its own instead of being traded against q_ref. 0 = off (unchanged behaviour).
    v4_qref_lowpass_hz: float = 0.0
    # 2026-11-05 candidate option (user: "heel strike 탁 할때 강성 탁 오르는거 처럼"; default off, no version bump): terminal-
    # swing pre-activation. Per leg, swing progress t_sw / T_sw (T_sw = running mean of the previous swing durations) ramps
    # the swing log-centre and half-width of the listed joints toward their stance values over v4_preact_p, so the
    # stiffness is already at the stance level at heel strike instead of one control step (33 ms) after the load is seen.
    # Basis: hamstrings active from ~85 % and vasti from ~95 % of the gait cycle (terminal swing, Perry & Burnfield 2010),
    # i.e. 50-100 ms before contact; the ramp window itself is an assumption.
    # 2026-11-05 candidate (user: "토크 맞출수 있게 보상 조정"): human joint-moment profile reward vic_moment_prior, off when
    # v4_moment_table is "". Table = Lencioni 2019 sagittal moments by speed and gait percent (training data; the W2-2 judge
    # is WBDS 2018). v4_moment_signs maps table signs to plant signs (+ hip extension, knee flexion, ankle plantarflexion).
    v4_moment_table: str = ""
    v4_moment_signs: tuple = (1.0, 1.0, 1.0)
    v4_moment_sd_floor: float = 0.1
    v4_moment_kernel: float = 1.5
    # 2026-11-06 candidate (user: "방법1 로 학습 바로 돌리자"; default off, no version bump): publish the same leg moments
    # (N m/kg, [N, 6] = L hip / knee / ankle, R hip / knee / ankle, plant signs) as ctx.steering.vic_moment_feat so the AMP
    # discriminator can see torque next to the body motion (protomotions/envs/obs/disc_moment.py).
    v4_moment_disc: bool = False
    v4_preact_joints: tuple = ()
    v4_preact_p: tuple = (0.8, 1.0)
    v4_swing_T0_s: float = 0.4
    gait_r_table: str = ""
    gait_r_speeds: tuple = (0.8, 1.0, 1.4)
    gait_r_window: int = 600
    gait_r_min_count: int = 20


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
        self._beta_hold_m = torch.tensor([1.0 if j in tuple(c.beta_hold_joints) else 0.0 for j in c.vic_joints], device=dev)
        assert set(c.beta_hold_joints) <= set(c.vic_joints), c.beta_hold_joints
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
        self._sway = torch.zeros(n, device=dev)
        self._pulse_err = torch.zeros(n, device=dev)            # vic_tc_v3 pulse_err_cost; zeros for v2 (populate_context)
        self._imp = torch.zeros(n, device=dev)                  # vic_tc_v4 impedance prior reward; zeros otherwise
        self._eff = torch.zeros(n, max(1, c.vic_efference_steps), 2 * len(c.vic_joints), device=dev)
        self._sim_ready = False
        # HC_VIC_GAIN_LOG=<path.npz> (evaluation, 2026-10-27, off by default): k_p, k_d, delta and leg state of the VIC
        # joints at every control step, written at exit (jitter analysis of the stiffness output).
        self._gain_log_path = os.environ.get("HC_VIC_GAIN_LOG")
        self._gain_log = []
        self._u_log = []                                       # vic_tc_v2: filtered co-activation level u [T, N, J]
        self._qref_log = []                                    # v4: leg q_ref before / after v4_qref_lowpass_hz [T, 2, N, J]
        if self._gain_log_path:
            atexit.register(self._dump_gain_log)
        if c.vic_law == "v3":
            self._init_v3(n, dev)
        elif c.vic_law == "v4":
            self._init_v4(n, dev, ranges)
        elif c.vic_law != "v2":
            raise ValueError(f"vic_law must be v2, v3 or v4, got {c.vic_law}")

    def _bv(self):
        """Per-joint beta [J]: the global beta, beta_hold on beta_hold_joints (2026-11-03)."""
        return self.beta * (1.0 - self._beta_hold_m) + float(self.config.beta_hold) * self._beta_hold_m

    def _init_v4(self, n, dev, ranges):
        c = self.config
        J = len(c.vic_joints)
        self._qref_lp = torch.zeros(n, J, device=dev)   # v4_qref_lowpass_hz state (leg q_ref, 2026-11-05)
        t = lambda v: torch.tensor(v, device=dev, dtype=torch.float)
        lb_lo, lb_hi = torch.zeros(J, 3, device=dev), torch.zeros(J, 3, device=dev)
        for i, j in enumerate(c.vic_joints):
            key = j[2:]
            for s in range(3):
                if key in ranges:
                    blo, bhi = ranges[key]["b"][s]
                else:
                    b0 = float(self._b_old[i])
                    blo, bhi = b0 / c.frontal_band, b0 * c.frontal_band
                lb_lo[i, s], lb_hi[i, s] = math.log(blo), math.log(bhi)
        self._v4_lk = self._lk_h.clone()                        # log centre [J, 3]
        self._v4_h = self._half - math.log(c.band_widen)        # log half-width of the human range [J, 3]
        for i, j in enumerate(c.vic_joints):
            if j[2:] not in ranges:
                self._v4_h[i] = math.log(c.frontal_band)
        self._v4_lb, self._v4_lb_lo, self._v4_lb_hi = 0.5 * (lb_lo + lb_hi), lb_lo, lb_hi
        alpha = dict(VIC4_ALPHA, **c.v4_alpha)
        self._v4_alpha = t([alpha.get(j[2:], 0.5) for j in c.vic_joints])
        self._v4_c = t([vic3_defaults(j[2:])["c"] if j[2:] in VIC3_PARAMS else 3.0 for j in c.vic_joints])
        self._v4_ankle = t([1.0 if j[2:] == "Ankle_y" else 0.0 for j in c.vic_joints])
        lo, hi = VIC4_ANKLE_LATE_K
        self._v4_late = (0.5 * (math.log(lo) + math.log(hi)), 0.5 * (math.log(hi) - math.log(lo)))
        self._v4_load = torch.zeros(n, 2, device=dev)
        self._v4_tst = torch.zeros(n, 2, device=dev)
        self._v4_T = torch.full((n, 2), float(c.v4_stance_T0_s), device=dev)
        self._v4_tsw = torch.zeros(n, 2, device=dev)                      # v4_preact: seconds in swing per leg
        self._v4_Tsw = torch.full((n, 2), float(c.v4_swing_T0_s), device=dev)
        self._v4_pre_m = t([1.0 if any(j.endswith(k) for k in c.v4_preact_joints) else 0.0 for j in c.vic_joints])
        self._v4_a = torch.zeros(n, J, device=dev)              # filtered action (activation-like)
        self._v4_u = torch.zeros(n, J, device=dev)              # co-contraction level for J2M / reflex
        self._v4_log = []
        self._pulse_err = torch.zeros(n, device=dev)
        self._eff = torch.zeros(n, max(1, c.vic_efference_steps), 2 * J, device=dev)
        model = getattr(self.env.simulator, "_human_joint_model", None)
        self._v4_model = model if c.v4_passive else None
        if c.v4_passive and model is None:
            print("[vic] v4: no plant passive model (_human_joint_model); k_pass = 0", flush=True)

    def _init_v3(self, n, dev):
        c = self.config
        if c.v3_u_map not in ("relu", "affine", "softplus") or c.v3_coact_cost not in ("torque", "unit", "activation") \
                or c.v3_load_source not in ("sim", "estimate") or c.v3_load_gate not in ("none", "stance"):
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
        self._v3_noise = torch.zeros(n, J, device=dev)         # coloured unit noise of the motor-noise term
        self._v3_pert = torch.zeros(n, J, device=dev)          # current perturbation pulse torque
        self._v3_pert_left = torch.zeros(n, J, device=dev)     # seconds left of the pulse
        self._v3_stance_t = torch.zeros(n, J, device=dev)      # seconds since this joint's leg touched down
        self._v3_qd0 = torch.zeros(n, J, device=dev)           # joint speed at pulse onset
        self._pulse_err = torch.zeros(n, device=dev)
        self._eff = torch.zeros(n, max(1, self.config.vic_efference_steps), 2 * J, device=dev)
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
        if self.config.v4_prior_substep:
            sim.vic_prior_kgap, sim.vic_prior_bgap = z.clone(), z.clone()
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
        if getattr(self, "_v4_log", None):
            v = torch.stack(self._v4_log[:len(a)], 0).float().numpy()             # [T, 10, N, J]
            extra.update(k_pass=v[:, 0], k_act=v[:, 1], w_stance=v[:, 2], w_stand=v[:, 3], tau_co=v[:, 4])
            if v.shape[1] >= 10:    # 2026-10-30 (videos): reference centre and dead zone of k_tot, human range of b_tot
                extra.update(k_ref=v[:, 5], k_lo=v[:, 6], k_hi=v[:, 7], b_lo=v[:, 8], b_hi=v[:, 9])
            if v.shape[1] >= 12:    # 2026-11-01 (A2 ffx): feed-forward residual torque tau_res (lowpassed, N m) and PD torque
                extra.update(tau_ff=v[:, 10], tau_pd=v[:, 11])
        if self._qref_log:
            q = torch.stack(self._qref_log[:len(a)], 0).numpy()                    # [T, 2, N, J]
            extra.update(qref_raw=q[:, 0], qref=q[:, 1])
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

    def pre_physics_extra_action(self, extra: Tensor, q_ref: Tensor, teacher_q_ref: Tensor = None):
        """Called by the env each control step before physics, with the raw extra outputs and the
        processed PD targets (common dof order). teacher_q_ref (2026-11-02, feed-forward distillation, v4 only): the
        PD targets the frozen fixed-PD teacher would set in the same state; _pre_physics_v4 then stores the torque the
        teacher would apply minus this step's PD torque as the feed-forward target (self.distill_target)."""
        c = self.config
        self._distill_q_ref = teacher_q_ref
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
        if c.vic_law == "v4":
            return self._pre_physics_v4(q_ref, unit, tau_res, contact, root_v, fresh)
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
        sim.vic_ff_prior[:, self._idx] = self._bv() * prior
        sim.vic_ff_res_target[:, self._idx] = tau_res
        reset_now = self.env.progress_buf <= 1
        if reset_now.any():
            sim.vic_ff_res[reset_now] = 0.0
        # Reward inputs.
        dev_ = (self._lk + delta) - lk_t
        self._band = (torch.relu(dev_.abs() - self._half[jj, state]) / math.log(2.0)).clamp(max=4.0).mean(-1)
        self._cocon = (torch.relu(delta / self._dmax.clamp_min(1e-6)) if c.coact_unit_cost else torch.relu(delta)).mean(-1)
        tau = kp * err - kd * qd + self._bv() * prior + sim.vic_ff_res[:, self._idx]
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
        if getattr(self.config, "vic_law", "v2") == "v4":
            return self._v4_u                                  # tau_co / tau_max
        return (self._delta / self._dmax.clamp_min(1e-6)).clamp(-1.0, 1.0)

    def _pre_physics_v3(self, extra, q_ref, unit, tau_res, state, jj, fresh, s):
        """vic_tc_v3 gains (module docstring); same feed-forward, reward inputs and logs as v2 otherwise."""
        c = self.config
        sim = self.env.simulator
        dt = self.env.dt
        # Co-contraction level u in [0, 1], activation-like filter as v2 (delta_tau_up_s / delta_tau_down_s).
        u_cmd = unit
        if c.v3_u_map == "relu":
            u = torch.relu(unit)
        elif c.v3_u_map == "softplus":
            u = (torch.nn.functional.softplus(8.0 * unit) / 8.0).clamp(max=1.0)
        else:
            u = 0.5 * (unit + 1.0)
        if c.delta_tau_up_s > 0 and c.delta_tau_down_s > 0:
            up = 1.0 - math.exp(-dt / c.delta_tau_up_s)
            down = 1.0 - math.exp(-dt / c.delta_tau_down_s)
            step = torch.where(u > self._delta, up, down) * (u - self._delta)
            self._delta = torch.where(fresh[:, None], u, self._delta + step)
            u = self._delta
        if c.v3_u_add != 0.0:
            u = (u + c.v3_u_add).clamp(0.0, 1.0)
        # Load term input: active torque of the previous control step (no algebraic loop).
        src = getattr(sim, "human_active_torques", None) if c.v3_load_source == "sim" else None
        if src is not None and src.shape[0] == len(fresh):
            tau_prev = src[:, self._idx].detach().abs()
        else:
            tau_prev = self._v3_tau_est.abs()
        tau_abs = tau_prev                                   # ungated |tau| (motor-noise scale)
        if c.v3_load_gate == "stance":
            tau_prev = tau_prev * (state >= 1)
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
        sim.vic_ff_prior[:, self._idx] = self._bv() * prior
        if c.v3_motor_noise > 0:
            a_n = math.exp(-dt / c.v3_motor_noise_tau_s)
            xi = torch.randn_like(self._v3_noise)
            self._v3_noise = torch.where(fresh[:, None], xi, a_n * self._v3_noise + math.sqrt(1 - a_n * a_n) * xi)
            tau_co = kc / self._v3_c
            tau_res = tau_res + c.v3_motor_noise * (tau_abs + tau_co) * self._v3_noise
        if c.v3_perturb_rate_hz > 0:
            self._v3_stance_t = torch.where((state >= 1) & ~fresh[:, None], self._v3_stance_t + dt,
                                            torch.zeros_like(self._v3_stance_t))
            ok = (state == 2) | ((state == 1) & (self._v3_stance_t <= c.v3_perturb_loading_s))
            self._v3_pert_left = (self._v3_pert_left - dt).clamp(min=0.0)
            start = ok & (self._v3_pert_left <= 0) & (torch.rand_like(self._v3_pert) < c.v3_perturb_rate_hz * dt)
            size = c.v3_perturb_amp * self._tau_max * (0.5 + 0.5 * torch.rand_like(self._v3_pert))
            sign = torch.where(torch.rand_like(self._v3_pert) < 0.5, -1.0, 1.0)
            self._v3_pert = torch.where(start, sign * size, self._v3_pert)
            self._v3_pert_left = torch.where(start, torch.full_like(self._v3_pert_left, c.v3_perturb_dur_s),
                                             self._v3_pert_left)
            live = (self._v3_pert_left > 0) & ok & ~fresh[:, None]
            self._v3_pert = torch.where(live, self._v3_pert, torch.zeros_like(self._v3_pert))
            tau_res = tau_res + self._v3_pert
            if c.pulse_err_cost:
                self._v3_qd0 = torch.where(start, qd, self._v3_qd0)
                e = (live * (qd - self._v3_qd0)) ** 2
                self._pulse_err = e.mean(-1).clamp(max=4.0)
        if c.vic_efference_steps > 0:
            now = torch.cat([u_cmd, tau_res / self._tau_max], -1)
            self._eff = torch.cat([now[:, None], self._eff[:, :-1]], 1)
            self._eff = torch.where(fresh[:, None, None], now[:, None].expand_as(self._eff), self._eff)
        sim.vic_ff_res_target[:, self._idx] = tau_res
        reset_now = self.env.progress_buf <= 1
        if reset_now.any():
            sim.vic_ff_res[reset_now] = 0.0
        if c.sway_cost:
            standing = (state == 2).all(-1)
            v = sim.get_robot_state().rigid_body_vel[:, 0, :2].norm(dim=-1)
            self._sway = torch.where(standing, (torch.relu(v - c.sway_v_h) / c.sway_v_h).clamp(max=4.0),
                                     torch.zeros_like(v))
        # Reward inputs: dead-zone guard (never toward the centre), co-contraction cost, effort, metabolic proxy.
        self._band = vic3_guard(lk, g_lo, g_hi)
        if c.v3_coact_cost == "torque":
            self._cocon = (kc / (self._v3_c * self._tau_max)).mean(-1)
        else:
            self._cocon = u.mean(-1)
        tau = kp * err - kd * qd + self._bv() * prior + sim.vic_ff_res[:, self._idx]
        if c.v3_coact_cost == "activation":
            r_net = (tau / self._tau_max).clamp(-2.0, 2.0)
            r_all = (tau.abs() + kc / self._v3_c) / self._tau_max
            self._cocon = (r_all.clamp(max=2.0) ** 2 - r_net ** 2).clamp(min=0.0, max=4.0).mean(-1)
        self._v3_tau_est = torch.maximum(torch.minimum(tau, self._tau_max), -self._tau_max).detach()
        self._effort = ((tau / self._tau_max) ** 2).clamp(max=4.0).mean(-1)
        p = (tau * qd)[:, self._sag]
        self._metab = (p.clamp(min=0).sum(-1) / 0.25 + (-p).clamp(min=0).sum(-1) / 1.20) / self._bw
        for st in range(3):
            m = state == st
            cnt = m.float().sum(0).clamp_min(1)
            self._mean_k[st] = torch.where(m.any(0), (lk * m).sum(0) / cnt, self._mean_k[st])
        self._log()

    def _v4_training_metrics(self, wl, q, err, qd, kp, kd, prior, fresh):
        """Training-time gait and hand-over metrics (2026-11-01, logging only; env/vic_gait/*, env/vic_support/*).

        Gait: per leg, at each swing -> stance switch (stance weight > 0.5), the knee flexion at strike and the largest knee
        flexion and ankle dorsiflexion of the swing that just ended (deg) -- the failure signature of the collapsed A2 runs
        (human about 3, 60 and +6 deg). Support: share of |beta prior| and |feed-forward| in |PD + prior + feed-forward| on
        the sagittal hip / knee / ankle, and alpha / beta, so the hand-over from the fixed PD is visible during training.
        2026-11-02 caveat: the stance weight comes from the low-passed load, so early swing still counts as stance and
        knee_swing_peak_deg misses the real peak (vic4c s5 control: 22 deg logged vs 51-57 deg in the HMV rollout); use
        the gait-% based knee_swing_peak{L,R}_vXX of _gait_r_metrics instead.
        """
        c = self.config
        names = list(self.config.vic_joints)
        if not hasattr(self, "_gm_knee"):
            self._gm_knee = torch.tensor([names.index("L_Knee_y"), names.index("R_Knee_y")], device=q.device)
            self._gm_ank = torch.tensor([names.index("L_Ankle_y"), names.index("R_Ankle_y")], device=q.device)
            self._gm_sag = torch.tensor([names.index(j) for j in ("L_Hip_y", "L_Knee_y", "L_Ankle_y",
                                                                  "R_Hip_y", "R_Knee_y", "R_Ankle_y")], device=q.device)
            self._gm_prev = torch.ones(q.shape[0], 2, dtype=torch.bool, device=q.device)
            self._gm_kmax = torch.zeros(q.shape[0], 2, device=q.device)
            self._gm_amax = torch.full((q.shape[0], 2), -1e3, device=q.device)
        knee, dorsi = torch.rad2deg(q[:, self._gm_knee]), -torch.rad2deg(q[:, self._gm_ank])
        stance = wl[:, :, 1] > 0.5
        f = fresh[:, None].expand_as(stance)
        self._gm_prev = torch.where(f, stance, self._gm_prev)
        self._gm_kmax = torch.where(f | stance, torch.zeros_like(knee), torch.maximum(self._gm_kmax, knee))
        self._gm_amax = torch.where(f | stance, torch.full_like(dorsi, -1e3), torch.maximum(self._gm_amax, dorsi))
        strike = stance & ~self._gm_prev & ~f
        out = {}
        if strike.any():
            out["vic_gait/knee_strike_deg"] = knee[strike].mean().reshape(1)
            out["vic_gait/knee_swing_peak_deg"] = self._gm_kmax_prev[strike].mean().reshape(1) \
                if hasattr(self, "_gm_kmax_prev") else knee[strike].mean().reshape(1)
            out["vic_gait/ankle_swing_dorsi_deg"] = self._gm_amax_prev[strike].mean().reshape(1) \
                if hasattr(self, "_gm_amax_prev") else dorsi[strike].mean().reshape(1)
        self._gm_prev = stance
        self._gm_kmax_prev, self._gm_amax_prev = self._gm_kmax.clone(), self._gm_amax.clone()
        s = self._gm_sag
        pd = (kp * err - kd * qd)[:, s].abs()
        pr = (self._bv() * prior)[:, s].abs() if not self.config.v4_prior_substep else \
            (self._bv() * ((self._k_old - kp) * err - (self._b_old - kd) * qd))[:, s].abs()
        ff = self.env.simulator.vic_ff_res[:, self._idx][:, s].abs()
        tot = (pd + pr + ff).sum(-1).clamp_min(1e-6)
        out["vic_support/prior_share"] = (pr.sum(-1) / tot).mean().reshape(1)
        out["vic_support/ff_share"] = (ff.sum(-1) / tot).mean().reshape(1)
        # 2026-11-02: per joint (sagittal hip / knee / ankle, both legs) mean |tau_ff|, |PD| (N m) and the fraction of
        # env-legs with tau_ff above 95 % of its cap (tau_res_scale * tau_max), so feed-forward growth and saturation show
        cap = (c.tau_res_scale * self._tau_max[s]).clamp_min(1e-6)
        for k, name in enumerate(("hip", "knee", "ankle")):
            cols = [k, k + 3]
            out[f"vic_support/ff_abs_{name}"] = ff[:, cols].mean().reshape(1)
            out[f"vic_support/pd_abs_{name}"] = pd[:, cols].mean().reshape(1)
            out[f"vic_support/ff_sat_{name}"] = (ff[:, cols] > 0.95 * cap[cols]).float().mean().reshape(1)
        out["vic_support/alpha"] = torch.tensor([self.alpha], device=q.device)
        out["vic_support/beta"] = torch.tensor([self.beta], device=q.device)
        if self.config.gait_r_table:
            out.update(self._gait_r_metrics(knee, dorsi))
        self._train_metrics = out

    def _gait_r_metrics(self, knee, dorsi):
        """Stride-averaged knee / ankle curves vs the Fukuchi table (config gait_r_*; env/vic_gait/*_r_vXX).

        2026-11-02: also per leg (the HMV-Walk judge uses the right leg; both-leg means hid a left / right asymmetry):
        knee_rL_vXX / knee_rR_vXX, ankle_rL / ankle_rR, and from the same per-leg curves the swing peak (max knee over
        gait % 50-100) and the knee at strike (gait % 0), which replace the load-based swing peak (its low-passed load
        missed early swing, 22 vs 51-57 deg in the HMV rollout)."""
        c, dev = self.config, knee.device
        if not hasattr(self, "_gr_sum"):
            t = torch.load(c.gait_r_table, map_location=dev, weights_only=False)
            si = [int((t["speeds"] - v).abs().argmin()) for v in c.gait_r_speeds]
            self._gr_ref = t["mean"][si].to(dev).float()                       # [S, 101, 2] knee, ankle (deg)
            self._gr_sum = torch.zeros(len(si), 101, 2, 2, device=dev)          # [S, pct, leg, joint]
            self._gr_cnt = torch.zeros(len(si), 101, 2, device=dev)
            self._gr_steps, self._gr_last = 0, {}
        now = self.env.progress_buf.float()[:, None]
        pct = (now - self._leg_onset) / self._leg_stride.clamp_min(1.)
        ok = (self._leg_strides >= 2) & (pct <= 1.0) & (self._leg_stride * self.env.dt <= c.max_stride_s)
        ok &= ((self.target_ratio() - 1).abs() < 0.05)[:, None]
        p = (pct.clamp(0, 1) * 100).round().long()
        val = torch.stack([knee, dorsi], -1)                                   # [N, 2 legs, 2 joints]
        for i, v in enumerate(c.gait_r_speeds):
            m = ok & ((self._tar_speed - v).abs() < 0.1)[:, None]
            for leg in (0, 1):
                ml = m[:, leg]
                if ml.any():
                    self._gr_sum[i, :, leg].index_add_(0, p[ml, leg], val[ml, leg])
                    self._gr_cnt[i, :, leg].index_add_(0, p[ml, leg], torch.ones_like(p[ml, leg], dtype=torch.float))
        self._gr_steps += 1
        if self._gr_steps >= c.gait_r_window:

            def corr(x, y):
                a, b = x - x.mean(0), y - y.mean(0)
                return (a * b).sum(0) / ((a * a).sum(0) * (b * b).sum(0)).sqrt().clamp_min(1e-9)

            for i, v in enumerate(c.gait_r_speeds):
                tag = f"v{int(round(v * 10)):02d}"
                both = self._gr_cnt[i].sum(-1)
                okb = both >= c.gait_r_min_count     # gait % 100 is rarely sampled (next onset resets it)
                if okb.sum() >= 90:
                    cur = self._gr_sum[i][okb].sum(1) / both[okb][:, None]
                    r = corr(cur, self._gr_ref[i][okb])
                    self._gr_last[f"vic_gait/knee_r_{tag}"] = r[0].reshape(1)
                    self._gr_last[f"vic_gait/ankle_r_{tag}"] = r[1].reshape(1)
                for leg, side in ((0, "L"), (1, "R")):
                    okl = self._gr_cnt[i, :, leg] >= c.gait_r_min_count
                    if okl.sum() < 90:
                        continue
                    cur = self._gr_sum[i, :, leg][okl] / self._gr_cnt[i, :, leg][okl][:, None]
                    r = corr(cur, self._gr_ref[i][okl])
                    self._gr_last[f"vic_gait/knee_r{side}_{tag}"] = r[0].reshape(1)
                    self._gr_last[f"vic_gait/ankle_r{side}_{tag}"] = r[1].reshape(1)
                    pk = torch.arange(101, device=dev)[okl]
                    kn = cur[:, 0]
                    self._gr_last[f"vic_gait/knee_swing_peak{side}_{tag}"] = kn[pk >= 50].max().reshape(1)
                    self._gr_last[f"vic_gait/knee_strike{side}_{tag}"] = kn[pk <= 1].mean().reshape(1) \
                        if (pk <= 1).any() else kn[:1]
            ks = [x for k, x in self._gr_last.items() if k.startswith("vic_gait/knee_r_v")]
            if ks:
                self._gr_last["vic_gait/knee_r_min"] = torch.cat(ks).min().reshape(1)
            kr = [x for k, x in self._gr_last.items() if k.startswith("vic_gait/knee_rR_v")]
            if kr:
                self._gr_last["vic_gait/knee_rR_min"] = torch.cat(kr).min().reshape(1)
            self._gr_sum.zero_(); self._gr_cnt.zero_(); self._gr_steps = 0
        return self._gr_last

    def log_values(self):
        """Per-step training metrics for env.extras (env/vic_gait/*, env/vic_support/*; v4 law only)."""
        return getattr(self, "_train_metrics", {})

    def _v4_leg_load(self, contact):
        """Per-leg vertical contact force / body weight [N, 2] (latest physics substep); contact flags as fallback."""
        sm = getattr(self.env.simulator, "_contact_sensor_map", None)
        if sm:
            try:
                f = torch.stack([sum(sm[b].data.net_forces_w_history[:, 0, 0, 2] for b in bodies)
                                 for bodies in self.config.grf_bodies], -1)
                return f.clamp(min=0.0) / self._bw
            except (KeyError, AttributeError, IndexError):
                pass
        return contact.float()

    def _pre_physics_v4(self, q_ref, unit, tau_res, contact, root_v, fresh):
        """vic_tc_v4 gains (module docstring); feed-forward, effort and metabolic inputs as v2."""
        c = self.config
        sim = self.env.simulator
        dt = self.env.dt
        J = len(c.vic_joints)
        f2 = fresh[:, None]
        if c.v4_freeze_a:
            unit = torch.zeros_like(unit)
        raw_log = q_ref[:, self._idx].float().clone() if self._gain_log_path else None
        if c.v4_qref_lowpass_hz > 0:     # in place: q_ref is the processed action the env passes to simulator.step
            raw = q_ref[:, self._idx]
            g = 1.0 - math.exp(-2.0 * math.pi * c.v4_qref_lowpass_hz * dt)
            self._qref_lp = torch.where(f2, raw, self._qref_lp + g * (raw - self._qref_lp))
            q_ref[:, self._idx] = self._qref_lp
        if raw_log is not None:      # 2026-11-05 (1-b evaluation): q_ref step-change RMS, filter cancelling check
            self._qref_log.append(torch.stack([raw_log, q_ref[:, self._idx].float()], 0).cpu())
        # Continuous leg-state weights.
        load = self._v4_leg_load(contact)
        g = 1.0 - math.exp(-dt / max(c.v4_load_tau_s, 1e-6))
        self._v4_load = torch.where(f2, load, self._v4_load + g * (load - self._v4_load))
        slow = ((2 * c.stand_speed - self._tar_speed) / c.stand_speed).clamp(0.0, 1.0) \
            * ((2 * c.stand_root_speed - root_v) / c.stand_root_speed).clamp(0.0, 1.0)
        wl = vic4_leg_weights(self._v4_load, slow, c.v4_load_lo, c.v4_load_hi)        # [N, 2, 3]
        w = wl[:, self._leg]                                                            # [N, J, 3]
        lk_tab = self._v4_lk.expand(len(fresh), J, 3).clone()
        h_tab = self._v4_h.expand(len(fresh), J, 3).clone()
        if c.v4_late_anchor:
            on = wl[:, :, 1] > 0.5
            ended = ~on & (self._v4_tst > 0.2)
            self._v4_T = torch.where(ended, 0.7 * self._v4_T + 0.3 * self._v4_tst, self._v4_T)
            self._v4_tst = torch.where(on & ~f2, self._v4_tst + dt, torch.zeros_like(self._v4_tst))
            p0, p1 = c.v4_late_p
            late = ((self._v4_tst / self._v4_T.clamp_min(0.2) - p0) / max(p1 - p0, 1e-6)).clamp(0.0, 1.0)
            ell = late[:, self._leg] * self._v4_ankle                                   # [N, J]
            lk_tab[..., 1] = (1 - ell) * lk_tab[..., 1] + ell * self._v4_late[0]
            h_tab[..., 1] = (1 - ell) * h_tab[..., 1] + ell * self._v4_late[1]
        if c.v4_preact_joints:
            sw = wl[:, :, 1] + wl[:, :, 2] < 0.5                                       # [N, 2] leg in swing
            done = ~sw & (self._v4_tsw > 0.15)
            self._v4_Tsw = torch.where(done, 0.7 * self._v4_Tsw + 0.3 * self._v4_tsw, self._v4_Tsw)
            self._v4_tsw = torch.where(sw & ~f2, self._v4_tsw + dt, torch.zeros_like(self._v4_tsw))
            q0, q1 = c.v4_preact_p
            rho = ((self._v4_tsw / self._v4_Tsw.clamp_min(0.15) - q0) / max(q1 - q0, 1e-6)).clamp(0.0, 1.0)
            rho = rho[:, self._leg] * self._v4_pre_m                                    # [N, J]
            lk_tab[..., 0] = (1 - rho) * lk_tab[..., 0] + rho * lk_tab[..., 1]
            h_tab[..., 0] = (1 - rho) * h_tab[..., 0] + rho * h_tab[..., 1]
        lk_c, h = (w * lk_tab).sum(-1), (w * h_tab).sum(-1)
        if c.v4_kc_speed_gamma:
            if getattr(self, "_v4_gam", None) is None:
                self._v4_gam = torch.tensor([float(next((g for k, g in c.v4_kc_speed_gamma.items() if j.endswith(k)), 0.0))
                                             for j in c.vic_joints], device=lk_c.device, dtype=lk_c.dtype)
                self._v4_vs = self._tar_speed.clone().to(lk_c.dtype)
            gv = 1.0 - math.exp(-dt / max(c.v4_kc_speed_tau_s, 1e-6))
            tv = self._tar_speed.to(lk_c.dtype)
            self._v4_vs = torch.where(fresh, tv, self._v4_vs + gv * (tv - self._v4_vs))
            lv = torch.log(self._v4_vs.clamp(*c.v4_kc_speed_clip) / c.v4_kc_speed_ref)
            kshift = w[..., 1] * self._v4_gam * lv[:, None]
            lk_c = lk_c + kshift
        lb_c = vic4_blend(w, self._v4_lb)
        lb_lo, lb_hi = vic4_blend(w, self._v4_lb_lo), vic4_blend(w, self._v4_lb_hi)
        if c.v4_kc_speed_gamma and c.v4_kc_speed_damp:
            # 2026-11-04: the damping centre and its clamp move by alpha_j * the stiffness-centre shift, the law's own
            # k-b coupling, so the speed scaling does not lower the damping ratio (zeta ~ b / sqrt(k)).
            bshift = self._v4_alpha * kshift
            lb_c, lb_lo, lb_hi = lb_c + bshift, lb_lo + bshift, lb_hi + bshift
        # Action: activation-like filter on a (as v2), then the total stiffness inside twice the human range.
        if c.delta_tau_up_s > 0 and c.delta_tau_down_s > 0:
            up = 1.0 - math.exp(-dt / c.delta_tau_up_s)
            down = 1.0 - math.exp(-dt / c.delta_tau_down_s)
            step = torch.where(unit > self._v4_a, up, down) * (unit - self._v4_a)
            self._v4_a = torch.where(f2, unit, self._v4_a + step)
        else:
            self._v4_a = unit
        delta = (h + math.log(c.v4_band_margin)) * self._v4_a
        lk_tot = lk_c + delta
        lb_tot = vic4_damping(lk_tot, lk_c, lb_c, self._v4_alpha, lb_lo, lb_hi)
        dof = sim._get_simulator_dof_state().convert_to_common(sim.data_conversion)
        q, qd = dof.dof_pos[:, self._idx], dof.dof_vel[:, self._idx]
        if self._v4_model is not None:
            k_pass, b_pass = vic4_passive_diag(self._v4_model, dof.dof_pos, self._idx)
            b_pass = b_pass.expand_as(k_pass)
        else:
            k_pass, b_pass = torch.zeros_like(q), torch.zeros_like(q)
        k_c = torch.exp(lk_c)
        k_act = torch.maximum(torch.exp(lk_tot) - k_pass, c.v4_k_floor_frac * k_c)
        b_act = torch.maximum(torch.exp(lb_tot) - b_pass, 0.01 * torch.exp(lb_c))
        a = self.alpha
        lk = (1 - a) * torch.log(self._k_old) + a * torch.log(k_act)
        lb = (1 - a) * torch.log(self._b_old) + a * torch.log(b_act)
        kp, kd = torch.exp(lk), torch.exp(lb)
        self._lk, self._lb = lk, lb
        sim._common_p_gains[:, self._idx] = kp
        sim._common_d_gains[:, self._idx] = kd
        self._write_gains()
        err = q_ref[:, self._idx] - q
        prior = (self._k_old - kp) * err - (self._b_old - kd) * qd
        tq = getattr(self, "_distill_q_ref", None)
        if tq is not None:      # feed-forward distillation target (control rate): teacher torque - this step's PD torque
            tau_t = self._k_old * (tq[:, self._idx] - q) - self._b_old * qd
            self.distill_target = tau_t - (kp * err - kd * qd + self._bv() * prior)
        # 2026-11-01 (A2 ff pre-fill, logging only): the torque the final v4 gains (alpha 1) lack against the fixed PD at
        # this state; a feed-forward that outputs it makes alpha 1 / beta 0 torque-equal to the fixed-PD walker.
        self._v4_dtau = (self._k_old - k_act) * err - (self._b_old - b_act) * qd
        self._v4_training_metrics(wl, q, err, qd, kp, kd, prior, fresh)
        if c.v4_prior_substep:
            sim.vic_prior_kgap[:, self._idx] = self._bv() * (self._k_old - kp)
            sim.vic_prior_bgap[:, self._idx] = self._bv() * (self._b_old - kd)
            sim.vic_ff_prior[:, self._idx] = 0.0
        else:
            sim.vic_ff_prior[:, self._idx] = self._bv() * prior
        sim.vic_ff_res_target[:, self._idx] = tau_res
        reset_now = self.env.progress_buf <= 1
        if reset_now.any():
            sim.vic_ff_res[reset_now] = 0.0
        # Co-contraction (J2M), impedance prior and costs; k_eff = what the joint really has (PD + passive).
        src = getattr(sim, "human_active_torques", None)
        tau_abs = src[:, self._idx].detach().abs() if src is not None and src.shape[0] == len(fresh) \
            else torch.zeros_like(q)
        tau_co, self._v4_u = vic4_coactivation(kp, self._v4_c, tau_abs, self._tau_max)
        k_eff, b_eff = kp + k_pass, kd + b_pass
        out = vic4_dead_zone(torch.log(k_eff), lk_c, h)
        self._band = (out / math.log(2.0)).clamp(max=4.0).mean(-1)
        if c.v4_imp_skip_swing_knee:
            m = torch.ones_like(out)
            kn = [i for i, j in enumerate(c.vic_joints) if j.endswith("Knee_y")]
            m[:, kn] = 1.0 - w[:, kn, 0]
            self._imp = torch.exp(-(m * (out / c.v4_imp_scale) ** 2).sum(-1) / m.sum(-1).clamp_min(1e-6))
        else:
            self._imp = torch.exp(-((out / c.v4_imp_scale) ** 2).mean(-1))
        self._cocon = self._v4_u.mean(-1)
        tau = kp * err - kd * qd + self._bv() * prior + sim.vic_ff_res[:, self._idx]
        self._effort = ((tau / self._tau_max) ** 2).clamp(max=4.0).mean(-1)
        pw = (tau * qd)[:, self._sag]
        self._metab = (pw.clamp(min=0).sum(-1) / 0.25 + (-pw).clamp(min=0).sum(-1) / 1.20) / self._bw
        state = w.argmax(-1)                                     # for logs only (dominant weight)
        if self._gain_log_path:
            self._gain_log.append(torch.stack([k_eff, b_eff, delta, state.float()], 0).half().cpu())
            self._u_log.append(self._v4_u.half().cpu())
            self._v4_log.append(torch.stack([k_pass, kp, w[..., 1], w[..., 2], tau_co, k_c, torch.exp(lk_c - h),
                                             torch.exp(lk_c + h), torch.exp(lb_lo), torch.exp(lb_hi),
                                             sim.vic_ff_res[:, self._idx], kp * err - kd * qd], 0).half().cpu())
            if len(self._gain_log) % 250 == 0:
                self._dump_gain_log()
        for st in range(3):
            m = state == st
            cnt = m.float().sum(0).clamp_min(1)
            self._mean_k[st] = torch.where(m.any(0), (torch.log(k_eff) * m).sum(0) / cnt, self._mean_k[st])
        self._log()

    def _leg_moments(self):
        """Total sagittal leg torque (active PD + feed-forward as applied, plus plant passive) in N m/kg, [N, 2, 3]
        (L, R x hip / knee / ankle); None before the simulator is ready."""
        c = self.config
        n = len(self._tar_speed)
        if not self._sim_ready:
            return None
        if getattr(self, "_mom_idx", None) is None:
            self._mom_idx = torch.tensor([[c.vic_joints.index(f"{l}_{j}") for j in ("Hip_y", "Knee_y", "Ankle_y")]
                                          for l in ("L", "R")], device=self._tar_speed.device)
        sim = self.env.simulator
        src = getattr(sim, "human_active_torques", None)
        if src is None or src.shape[0] != n:
            return None
        tau = src[:, self._idx].float()
        if self._v4_model is not None:
            dof = sim._get_simulator_dof_state().convert_to_common(sim.data_conversion)
            tau = tau + vic4_passive_torque(self._v4_model, dof.dof_pos, self._idx)
        return tau[:, self._mom_idx] / c.body_mass                                      # [N, 2, 3]

    def _moment_prior(self, ctx, tau):
        """vic_moment_prior from the leg moments; ones when the option is off or before the simulator is ready."""
        c = self.config
        if not c.v4_moment_table or tau is None:
            return torch.ones(len(self._tar_speed), device=self._tar_speed.device)
        return vic_moment_prior(tau, self._tar_speed, ctx.steering.gait_pct, ctx.steering.gait_pct_valid,
                                c.v4_moment_table, c.v4_moment_signs, c.v4_moment_sd_floor, c.v4_moment_kernel)

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
        ctx.steering.vic_sway = self._sway
        ctx.steering.vic_pulse_err = self._pulse_err
        ctx.steering.vic_efference = self._eff.flatten(1)
        ctx.steering.vic_imp_prior = self._imp
        tau = self._leg_moments() if (self.config.v4_moment_table or self.config.v4_moment_disc) else None
        ctx.steering.vic_moment_prior = self._moment_prior(ctx, tau)
        if self.config.v4_moment_disc:
            ctx.steering.vic_moment_feat = (tau.flatten(1) if tau is not None
                                            else torch.zeros(len(self._tar_speed), 6, device=self._tar_speed.device))
