"""Offline check of the human_model_v4.0 ankle SEE against human walking data (CPU, no simulator).

Drives the SEE with WBDS (Fukuchi 2018) mean ankle angle and internal ankle moment at 0.8 / 1.0 / 1.4 m/s
(hmv kit human_ref: fukuchi2018_knee_ankle_by_speed.pt, wbds2018_sagittal_moment_by_speed_plantbsp_v35.pt; 63.3 kg):
  quasi-static   the whole plantarflexor moment goes through the Achilles channel: D = T^-1(M), phi = q + D
  module replay  common/series_elastic.py at 240 Hz with the joint angle prescribed, Kp = Kd = 0, ff = M (CE torque = M),
                 tendon damping on (tau_T 3 ms): checks the integrated code and gives the hysteresis loss
and evaluates the v3.6 plantarflexion cap (Anderson 2007 Hill curve, v3.6 young-adult scale) at the joint velocity
(v3.6) and at the CE velocity (v4.0). Stride times are assumptions (only velocities / power / damping loss depend on them).

usage: PYTHONPATH=<ProtoMotions> python -m protomotions.robot_configs.human_model.human_model_v4_0.validate_see [out.json]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from ..common.series_elastic import SeriesElasticAnkle, tendon_energy, tendon_stretch
from .model_config import series_elastic, ACHILLES, active_strength_scale, passive_exponentials

HUMAN_REF = Path("/home/ghlee/scratch/hmv_kit_b3bb6ef2/human_ref")
BODY_MASS = 63.3
STRIDE_S = {0.8: 1.24, 1.0: 1.14, 1.4: 1.03}     # assumption: typical adult cadence 97 / 105 / 117 steps/min
STANCE = slice(0, 62)                              # 0-61 % of the cycle (toe-off ~60-62 %)
MIDSTANCE = slice(10, 46)                          # loading of the tendon (ankle dorsiflexes under rising moment)
PUSHOFF = slice(40, 62)


def human(speed):
    m = torch.load(HUMAN_REF / "wbds2018_sagittal_moment_by_speed_plantbsp_v35.pt", weights_only=False)
    a = torch.load(HUMAN_REF / "fukuchi2018_knee_ankle_by_speed.pt", weights_only=False)
    i = int((m["speeds"] - speed).abs().argmin())
    M = m["mean"][i, :, 2].double().numpy() * BODY_MASS                 # N m, + plantarflexion (internal)
    q = -np.deg2rad(a["mean"][i, :, 1].double().numpy())                 # plant Ankle_y, + plantarflexion
    knee = np.deg2rad(a["mean"][i, :, 0].double().numpy())               # + flexion
    return M, q, knee, int(m["n_trials"][i])


def caps_pf(q_ankle, knee, qd_ankle):
    """v3.6 plantarflexion cap at (ankle angle, knee angle, ankle velocity) [T]."""
    from ..common.dynamics import HumanJointModel
    from ..common.profile import load_profile
    prof = load_profile("human_model_v3")
    names = list(prof["joints"])
    hm = HumanJointModel(prof, names, "cpu", torch.float64, active_strength_scale=active_strength_scale(),
                         passive_exponentials=passive_exponentials())
    T = len(q_ankle)
    q = torch.zeros(T, len(names), dtype=torch.float64)
    qd = torch.zeros_like(q)
    for side in ("L", "R"):
        q[:, names.index(f"{side}_Ankle_y")] = torch.from_numpy(q_ankle)
        q[:, names.index(f"{side}_Knee_y")] = torch.from_numpy(knee)
        qd[:, names.index(f"{side}_Ankle_y")] = torch.from_numpy(qd_ankle)
    q = torch.maximum(torch.minimum(q, hm.upper), hm.lower)
    pos = hm.strength_caps(q, qd)[1]
    return pos[:, names.index("L_Ankle_y")].numpy()


def quasi_static(speed, k_tendon=None):
    cfg = series_elastic(achilles=None if k_tendon is None else {"k_tendon_n_per_m": k_tendon})
    pf = cfg["channels"]["PF"]
    k, toe, r, L = pf["k_nm_per_rad"], pf["toe_rad"], pf["moment_arm_m"], pf["slack_length_m"]
    M, q, knee, n = human(speed)
    t = np.linspace(0, STRIDE_S[speed], 101)
    dt = t[1] - t[0]
    F = np.clip(M, 0, None)
    D = tendon_stretch(torch.from_numpy(F), k, toe).numpy()
    E = tendon_energy(torch.from_numpy(D), k, toe).numpy()
    phi = q + D
    qd, phid, Dd = np.gradient(q, dt), np.gradient(phi, dt), np.gradient(D, dt)
    Pj, Pce, Pt = M * qd, F * phid, -F * Dd                  # joint, CE (muscle), tendon -> joint power
    w = lambda p, s: float(np.trapz(np.clip(p[s], 0, None), t[s]))
    W_joint_pos = w(Pj, STANCE)
    W_joint_push = w(Pj, PUSHOFF)
    returned = w(Pt, STANCE)
    stored = w(-Pt, STANCE)
    cap_joint = caps_pf(q, knee, qd)
    cap_ce = caps_pf(q, knee, phid)
    po = PUSHOFF
    return dict(
        speed_mps=speed, wbds_trials=n, stride_s_assumed=STRIDE_S[speed],
        peak_moment_nm=round(float(M.max()), 1), peak_moment_nm_per_kg=round(float(M.max() / BODY_MASS), 3),
        peak_tendon_force_n=round(float(F.max() / r)), peak_tendon_force_bw=round(float(F.max() / r / (BODY_MASS * 9.81)), 2),
        peak_elongation_mm=round(float(D.max() * r * 1000), 1), peak_strain_pct=round(float(D.max() * r / L * 100), 2),
        peak_stored_energy_j=round(float(E.max()), 2), tendon_stored_j=round(stored, 2), tendon_returned_j=round(returned, 2),
        ankle_positive_work_stance_j=round(W_joint_pos, 2), ankle_positive_work_pushoff_j=round(W_joint_push, 2),
        ce_positive_work_stance_j=round(w(Pce, STANCE), 2),
        tendon_share_of_ankle_positive_work_pct=round(100 * returned / W_joint_pos, 1),
        midstance_mtu_excursion_mm=round(float(r * (q[MIDSTANCE].max() - q[MIDSTANCE].min()) * 1000), 1),
        midstance_ce_excursion_mm=round(float(r * (phi[MIDSTANCE].max() - phi[MIDSTANCE].min()) * 1000), 1),
        peak_pf_velocity_joint_rad_s=round(float(qd[STANCE].max()), 2), peak_pf_velocity_ce_rad_s=round(float(phid[STANCE].max()), 2),
        pushoff_need_over_cap_joint_velocity=round(float((F[po] / cap_joint[po]).max()), 3),
        pushoff_need_over_cap_ce_velocity=round(float((F[po] / cap_ce[po]).max()), 3),
    ), (t, M, q, D)


def replay(speed, fps=240):
    """Run the SEE module itself (two cycles, report the second) with the joint angle prescribed."""
    cfg = series_elastic()
    cfg["joints"] = ["A"]
    see = SeriesElasticAnkle(cfg, ["A"], 1, "cpu", torch.float64)
    M, q, _, _ = human(speed)
    T = STRIDE_S[speed]
    t = np.arange(0, 2 * T, 1 / fps)
    x = (t / T) % 1.0
    pct = np.linspace(0, 1, 101)
    Mi, qi = np.interp(x, pct, M), np.interp(x, pct, q)
    big = torch.full((1, 1), 1e4, dtype=torch.float64)
    zero = torch.zeros(1, 1, dtype=torch.float64)
    rows = []
    for k in range(len(t)):
        qk = torch.tensor([[qi[k]]], dtype=torch.float64)
        qd = torch.tensor([[(qi[k] - qi[k - 1]) * fps if k else 0.0]], dtype=torch.float64)
        tau = see.substep(qk, qd, qk, zero, zero, torch.tensor([[Mi[k]]], dtype=torch.float64), big, big, 1 / fps)
        rows.append((t[k], Mi[k], tau.item(), see.D["PF"].item(), see.phid["PF"].item(), qd.item()))
    o = np.array(rows)
    o = o[o[:, 0] >= T]
    pf = cfg["channels"]["PF"]
    D = o[:, 3]
    Dd = np.gradient(D, 1 / fps)
    loss = float(np.sum(pf["damping_nms_per_rad"] * Dd ** 2) / fps)
    Emax = float(tendon_energy(torch.from_numpy(D), pf["k_nm_per_rad"], pf["toe_rad"]).max())
    return dict(speed_mps=speed, fps=fps, max_abs_joint_torque_minus_human_moment_nm=round(float(np.abs(o[:, 2] - o[:, 1]).max()), 4),
                peak_strain_pct=round(float(D.max() * pf["moment_arm_m"] / pf["slack_length_m"] * 100), 2),
                damping_loss_j=round(loss, 3), damping_loss_frac_of_peak_stored=round(loss / Emax, 3))


def standing():
    """Quiet standing: tendon tangent stiffness per leg at the standing calf torque vs the gravitational stiffness."""
    pf = series_elastic()["channels"]["PF"]
    k, toe = pf["k_nm_per_rad"], pf["toe_rad"]
    com_h, lever = 0.90, 0.045            # assumptions: CoM height above the ankle (m), CoM ahead of the ankle (m)
    tau = 0.5 * BODY_MASS * 9.81 * lever
    D = float(tendon_stretch(torch.tensor(tau, dtype=torch.float64), k, toe))
    kt = k * min(D / toe, 1.0)
    mgh = BODY_MASS * 9.81 * com_h
    return dict(standing_calf_torque_per_leg_nm=round(tau, 1), tendon_tangent_stiffness_per_leg_nm_per_rad=round(kt, 1),
                both_legs_tendon_over_mgh=round(2 * kt / mgh, 2),
                note="upper bound of intrinsic ankle stiffness (CE in series lowers it); Loram & Lakie 2002 J Physiol "
                     "545:1041 measured intrinsic stiffness 91 +- 23 % of mgh (memory, not re-verified here)")


def main(out=None):
    res = dict(model=series_elastic(), quasi_static=[], replay=[], sensitivity_k_tendon_1p0=[], standing=standing())
    for v in (0.8, 1.0, 1.4):
        res["quasi_static"].append(quasi_static(v)[0])
        res["replay"].append(replay(v))
    for k in (145e3, 180e3, 231e3):
        r = quasi_static(1.0, k)[0]
        res["sensitivity_k_tendon_1p0"].append(dict(k_tendon_n_per_mm=k / 1e3, **{x: r[x] for x in (
            "peak_strain_pct", "peak_elongation_mm", "tendon_share_of_ankle_positive_work_pct", "midstance_ce_excursion_mm",
            "pushoff_need_over_cap_ce_velocity")}))
    res["literature"] = {
        "peak_strain_walking_pct": "4.6 (10.9 mm), ~5 km/h, Lichtwark & Wilson 2006 J Exp Biol 209:4379",
        "tendon_stretch_single_support_mm": "~7, 3 km/h, Fukunaga 2001 Proc R Soc B 268:229",
        "peak_at_force_walking_n": "1430 +- 500 (optic fibre, 1.1-1.8 m/s, Finni 1998); 2200-2700 (inverse dynamics, Zelik & Franz 2017)",
        "fascicles": "MG near-isometric in stance (Fukunaga 2001; Lichtwark & Wilson 2006 57.7 +- 2.3 mm); soleus lengthens to end of single stance (Ishikawa 2005)",
        "ankle_positive_work_stance": "ankle 60 % of stance positive joint work, apparent efficiency ~0.61 from Achilles recoil (Sawicki 2009)",
        "hysteresis": "5-10 % (Peltonen 2013 5 %; Zelik & Franz 2017 2-11 % at 1.25 m/s)",
    }
    text = json.dumps(res, indent=1)
    if out:
        Path(out).write_text(text + "\n")
    return res


if __name__ == "__main__":
    r = main(sys.argv[1] if len(sys.argv) > 1 else None)
    keys = ("speed_mps", "peak_moment_nm_per_kg", "peak_tendon_force_n", "peak_elongation_mm", "peak_strain_pct",
            "tendon_stored_j", "tendon_returned_j", "ankle_positive_work_stance_j", "ce_positive_work_stance_j",
            "tendon_share_of_ankle_positive_work_pct", "midstance_mtu_excursion_mm", "midstance_ce_excursion_mm",
            "peak_pf_velocity_joint_rad_s", "peak_pf_velocity_ce_rad_s", "pushoff_need_over_cap_joint_velocity",
            "pushoff_need_over_cap_ce_velocity")
    for row in r["quasi_static"]:
        print({k: row[k] for k in keys})
    for row in r["replay"] + r["sensitivity_k_tendon_1p0"] + [r["standing"]]:
        print(row)
