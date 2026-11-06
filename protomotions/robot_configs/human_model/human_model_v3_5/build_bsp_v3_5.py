"""human_model_v3.5 segment inertia (2026-11-07, user-approved version: "플랜트 질량, 관성모멘트가 뭐가 안맞는거일 수 도 있으니
이 부분도 점검해서 업뎃필요하면 버젼 새버젼 해서 업뎃해줘").

Why: the v2 / v3.x plant has the SMPL shape-derived leg mass distribution: shank 5.6 % and foot (talus + foot + toe) 3.2 % of
body mass against 4.6 % / 1.3 % (de Leva 1996, mean of the male and female tables), thigh 11.7 % against 14.5 %, and every
leg segment centre of mass at mid-segment (de Leva: thigh 38.5 %, shank 44.4 % from the proximal joint). The leg moment of
inertia about the hip is about 1.6 times the human value, so the same swing needs much more hip torque (WBDS inverse dynamics
on the same trials: plant segment terms add up to +0.38 N m/kg of swing hip moment at 1.0 m/s). Arm radii of gyration were
0.11-0.14 of the segment length against 0.26-0.28.

What changes (mass properties only; geometry, colliders, joints, ROM, strength, PD and passive terms stay v3.4):
- thigh (Hip body), shank (Knee body), upper arm (Shoulder body), forearm (Elbow body): de Leva sex-averaged mass fraction,
  centre of mass on the joint-to-joint line, radii of gyration about the sagittal / transverse / longitudinal axes;
- foot (Talus + Ankle + Toe bodies) and hand (Wrist + Hand bodies): masses and inertias scaled to the de Leva fraction, centres
  of mass kept (the foot centre of mass is already 42 % along ankle-to-toe joints, de Leva 42 % heel-to-toe);
- trunk (Pelvis, Torso, Spine, Chest): masses and inertias scaled so the total mass stays 63.31 kg (body-weight normalised
  references are unchanged). Head, neck and clavicle (Thorax) bodies unchanged.
Body frames of this plant are world-aligned at the zero pose (USD bodies carry world transforms, MJCF bodies have no rotation),
so MJCF ipos / fullinertia and USD centerOfMass / principalAxes + diagonalInertia are the same numbers.

  python build_bsp_v3_5.py   (writes assets/human_model_v3_5.usda, assets/human_model_v3_5.xml, bsp_v3_5.json)
"""
import json
import re
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
SRC_XML = HERE.parent / "human_model_v2/assets/human_model_v2.xml"
# de Leva 1996 Table 4 (male, female): mass %, centre of mass % from the proximal joint, radii of gyration % (sag, trans, long)
DE_LEVA = dict(thigh=((14.16, 40.95, 32.9, 32.9, 14.9), (14.78, 36.12, 36.9, 36.4, 16.2)),
               shank=((4.33, 44.59, 25.5, 24.9, 10.3), (4.81, 44.16, 27.1, 26.7, 9.3)),
               foot=((1.37, 44.15, 25.7, 24.5, 12.4), (1.29, 40.14, 29.9, 27.9, 13.9)),
               upperarm=((2.71, 57.72, 28.5, 26.9, 15.8), (2.55, 57.54, 27.8, 26.0, 14.8)),
               forearm=((1.62, 45.74, 27.6, 26.5, 12.1), (1.38, 45.59, 26.1, 25.7, 9.4)),
               hand=((0.61, 79.00, 62.8, 51.3, 40.1), (0.56, 74.74, 53.1, 45.4, 33.5)))
AVG = {k: tuple((a + b) / 200.0 for a, b in zip(*v)) for k, v in DE_LEVA.items()}
LINE = dict(thigh=("Hip", "Hip_y", "Knee_y"), shank=("Knee", "Knee_y", "Ankle_y"),
            upperarm=("Shoulder", "Shoulder_y", "Elbow_y"), forearm=("Elbow", "Elbow_y", "Wrist_y"))
SCALED = dict(foot=("Talus", "Ankle", "Toe"), hand=("Wrist", "Hand"))
TRUNK = ("Pelvis", "Torso", "Spine", "Chest")


def quat_from_matrix(R):
    q = np.empty(4)
    t = np.trace(R)
    if t > 0:
        s = 2 * np.sqrt(t + 1); q[:] = (0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s)
    else:
        i = int(np.argmax(np.diag(R))); j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * np.sqrt(1 + R[i, i] - R[j, j] - R[k, k])
        q[0] = (R[k, j] - R[j, k]) / s; q[1 + i] = 0.25 * s; q[1 + j] = (R[j, i] + R[i, j]) / s; q[1 + k] = (R[k, i] + R[i, k]) / s
    return q / np.linalg.norm(q) * np.sign(q[0] if q[0] != 0 else 1)


def body_inertia_matrix(m, b):
    R = np.zeros(9); mujoco.mju_quat2Mat(R, m.body_iquat[b]); R = R.reshape(3, 3)
    return R @ np.diag(m.body_inertia[b]) @ R.T


def build():
    m = mujoco.MjModel.from_xml_path(str(SRC_XML)); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    assert np.allclose(d.xquat[1:], [1, 0, 0, 0], atol=1e-9), "body frames are not world-aligned"
    M = float(m.body_mass.sum())
    new = {}                                                  # body -> (mass, com [3], inertia [3x3] about com, body frame)
    for side in ("L", "R"):
        for seg, (body, pj, dj) in LINE.items():
            frac, cm, rs, rt, rl = AVG[seg]
            b = m.body(f"{side}_{body}").id
            P, D = d.xanchor[m.joint(f"{side}_{pj}").id], d.xanchor[m.joint(f"{side}_{dj}").id]
            L = float(np.linalg.norm(D - P)); u = (D - P) / L
            fwd = np.array([1.0, 0.0, 0.0]) if seg in ("thigh", "shank") else np.array([0.0, 0.0, 1.0])
            e1 = fwd - (fwd @ u) * u; e1 /= np.linalg.norm(e1); e2 = np.cross(u, e1)
            mass = frac * M
            I = mass * L ** 2 * (rs ** 2 * np.outer(e1, e1) + rt ** 2 * np.outer(e2, e2) + rl ** 2 * np.outer(u, u))
            new[f"{side}_{body}"] = (mass, P + cm * (D - P) - d.xpos[b], I, dict(length_m=L, de_leva=AVG[seg]))
        for seg, bodies in SCALED.items():
            ids = [m.body(f"{side}_{x}").id for x in bodies]
            f = AVG[seg][0] * M / m.body_mass[ids].sum()
            for b in ids:
                new[m.body(b).name] = (m.body_mass[b] * f, m.body_ipos[b].copy(), body_inertia_matrix(m, b) * f, dict(scale=f))
    changed = sum(v[0] for v in new.values()) - sum(m.body_mass[m.body(n).id] for n in new)
    tid = [m.body(x).id for x in TRUNK]
    ft = (m.body_mass[tid].sum() - changed) / m.body_mass[tid].sum()
    for b in tid:
        new[m.body(b).name] = (m.body_mass[b] * ft, m.body_ipos[b].copy(), body_inertia_matrix(m, b) * ft, dict(scale=ft))
    total = sum(v[0] for v in new.values()) + sum(m.body_mass[i] for i in range(m.nbody) if m.body(i).name not in new)
    assert abs(total - M) < 1e-6, (total, M)
    return m, M, new


def usd_block(name, mass, com, I):
    w, V = np.linalg.eigh(I)
    if np.linalg.det(V) < 0: V[:, 2] *= -1
    q = quat_from_matrix(V)
    f = lambda x: ", ".join(f"{v:.9g}" for v in x)
    return (f'        over "{name}"\n        {{\n            point3f physics:centerOfMass = ({f(com)})\n'
            f'            float3 physics:diagonalInertia = ({f(w)})\n            float physics:mass = {mass:.9g}\n'
            f'            quatf physics:principalAxes = ({f(q)})\n        }}\n')


def main():
    m, M, new = build()
    usd = ("#usda 1.0\n(\n    defaultPrim = \"smpl_humanoid\"\n    subLayers = [\n        @../../human_model_v3_1/assets/human_model_v3_1.usda@\n"
           "    ]\n    upAxis = \"Z\"\n)\n\n# human_model_v3.5 segment mass properties (build_bsp_v3_5.py); everything else is human_model_v3.1.\n"
           "over \"smpl_humanoid\"\n{\n    over \"bodies\"\n    {\n")
    usd += "".join(usd_block(n, *new[n][:3]) for n in sorted(new))
    usd += "    }\n}\n"
    (HERE / "assets/human_model_v3_5.usda").write_text(usd)
    xml = SRC_XML.read_text()
    for n, (mass, com, I, _) in new.items():
        full = f'<inertial pos="{com[0]:.9g} {com[1]:.9g} {com[2]:.9g}" mass="{mass:.9g}" fullinertia="{I[0,0]:.9g} {I[1,1]:.9g} {I[2,2]:.9g} {I[0,1]:.9g} {I[0,2]:.9g} {I[1,2]:.9g}" />'
        i = xml.index(f'<body name="{n}"'); e = xml.index(">", i) + 1
        own, depth = -1, 0                                     # this body's own <inertial> (depth 0), if any
        for t in re.finditer(r"<body |</body>|<inertial ", xml[e:]):
            if t.group() == "<body ": depth += 1
            elif t.group() == "</body>":
                if depth == 0: break
                depth -= 1
            elif depth == 0: own = e + t.start(); break
        if own != -1:
            xml = xml[:own] + full + xml[xml.index("/>", own) + 2:]
        else:
            xml = xml[:e] + "\n" + " " * 8 + full + xml[e:]
    (HERE / "assets/human_model_v3_5.xml").write_text(xml)
    m2 = mujoco.MjModel.from_xml_path(str(HERE / "assets/human_model_v3_5.xml"))
    assert abs(m2.body_mass.sum() - M) < 1e-5
    for n, (mass, com, I, _) in new.items():                   # the compiled model carries exactly the new values
        b = m2.body(n).id
        assert abs(m2.body_mass[b] - mass) < 1e-6 and np.allclose(m2.body_ipos[b], com, atol=1e-6), n
        assert np.allclose(body_inertia_matrix(m2, b), I, atol=1e-7), n
    rec = {n: dict(mass_kg=new[n][0], mass_pct=100 * new[n][0] / M, com_body_m=list(map(float, new[n][1])),
                   inertia_about_com=np.asarray(new[n][2]).round(7).tolist(), old_mass_kg=float(m.body_mass[m.body(n).id]), **new[n][3])
           for n in sorted(new)}
    json.dump(dict(total_mass_kg=M, de_leva_sex_mean=AVG, bodies=rec), open(HERE / "bsp_v3_5.json", "w"), indent=1, default=float)
    print("written; total mass", round(M, 3))


if __name__ == "__main__":
    main()
