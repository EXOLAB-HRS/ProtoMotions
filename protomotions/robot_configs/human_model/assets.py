# SPDX-License-Identifier: Apache-2.0
"""Validate the authored USDA and derive MJCF for engines that require it."""

import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import xml.etree.ElementTree as ET


def render_mjcf(source, profile, backend_limits):
    tree = ET.fromstring(source)
    compiler = tree.find("compiler")
    if compiler is not None and compiler.get("angle", "degree") != "degree":
        raise ValueError("Expected degree-based SMPL MJCF")
    joints = {e.get("name"): e for e in tree.findall(".//joint") if e.get("type") == "hinge"}
    if set(joints) != set(profile["joints"]):
        raise ValueError("MJCF/profile joint mismatch; refusing to adapt a different skeleton")
    for name, row in profile["joints"].items():
        joint = joints[name]
        expected = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[name[-1]]
        if list(map(float, joint.get("axis", "0 0 1").split())) != expected:
            raise ValueError(f"Unexpected SMPL axis convention: {name}")
        joint.set("range", " ".join(format(x, ".12g") for x in row["rom_deg"]))
        joint.set("limited", "true")
        # The original 300-1000/30-100 values were control gains. Tissue forces
        # now come from HumanJointModel, exactly once on every physics step.
        joint.set("stiffness", "0")
        joint.set("damping", "0")
        joint.set("frictionloss", "0")
        # Combined active+passive ceiling; active caps are direction-dependent
        # and enforced by the runtime before summing passive forces.
        limit = backend_limits[name]
        joint.set("actuatorfrcrange", f"{-limit:.12g} {limit:.12g}")
    motors = tree.findall("./actuator/motor")
    if {e.get("joint") for e in motors} != set(joints) or len(motors) != len(joints):
        raise ValueError("Expected one SMPL motor per hinge joint")
    for motor in motors:
        limit = backend_limits[motor.get("joint")]
        motor.set("gear", "1")
        motor.set("ctrllimited", "true")
        motor.set("ctrlrange", f"{-limit:.12g} {limit:.12g}")
    tree.insert(0, ET.Comment(" Locally derived human joint properties; see robot_configs/human_model/README.md "))
    return ET.tostring(tree, encoding="unicode") + "\n"


def validate_usda_rom(source, profile):
    """Check the committed RoM without changing or regenerating the USDA.

    JSON retains the same bounds for action limits and the MJCF backends.
    Reject disagreement instead of silently overwriting the authored asset.
    """
    if source.startswith("version https://git-lfs.github.com/spec"):
        raise ValueError("Human model USDA must be the actual text asset, not a Git LFS pointer")
    if not source.startswith("#usda"):
        raise ValueError("Expected the text human model USDA asset")
    seen = set()
    for match in re.finditer(r'        def PhysicsJoint "([^"]+)" \([\s\S]*?\n        }', source):
        block, body = match.group(0), match.group(1)
        for axis in "xyz":
            name = f"{body}_{axis}"
            row = profile["joints"].get(name)
            if row is None:
                raise ValueError(f"Unexpected USD joint: {name}")
            token = f"rot{axis.upper()}"
            if f'custom token mjcf:{token}:name = "{name}"' not in block:
                raise ValueError(f"USD joint/axis mapping changed: {name}")
            for bound, value in zip(("low", "high"), row["rom_deg"]):
                prop = f"limit:{token}:physics:{bound}"
                values = re.findall(rf"(?m)^\s*float {re.escape(prop)} = ([^\n]+)$", block)
                if len(values) != 1:
                    raise ValueError(f"Expected exactly one USD property {body}/{prop}")
                if not math.isclose(float(values[0]), value, rel_tol=1e-6, abs_tol=1e-6):
                    raise ValueError(f"USD/profile ROM mismatch: {name}; update the USDA and profile bounds together")
            if name in seen:
                raise ValueError(f"Duplicate USD joint axis: {name}")
            seen.add(name)
    if seen != set(profile["joints"]):
        raise ValueError("USD/profile joint mismatch")


def _atomic_write(path, text):
    # Content-addressed paths + atomic replacement support multiple env workers.
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        staging = Path(f.name)
        try:
            f.write(text)
            f.flush()
            os.replace(staging, path)
        finally:
            staging.unlink(missing_ok=True)


def derive_mjcf(xml_path, profile, backend_limits):
    xml_path = Path(xml_path)
    xml = xml_path.read_text()
    digest = hashlib.sha256()
    for data in (xml, json.dumps(profile, sort_keys=True), json.dumps(backend_limits, sort_keys=True), Path(__file__).read_text()):
        digest.update(data.encode())
    root = Path(os.environ.get("PROTOMOTIONS_HUMAN_MODEL_CACHE", str(Path(tempfile.gettempdir()) / f"protomotions-human-model-{os.getuid()}")))
    destination = root / digest.hexdigest()[:24]
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    derived_xml = destination / "smpl_humanoid.xml"
    if not derived_xml.exists():
        _atomic_write(derived_xml, render_mjcf(xml, profile, backend_limits))
    return destination


def derive_serial_usda(usd_path, *, frame_mass=1e-6):
    """Candidate XYZ hinge chain preserving the MJCF per-axis coordinates.

    PhysX needs positive inertial properties on intermediate links. Each of the
    46 invisible frames adds frame_mass kg and 1e-3*frame_mass kg m² isotropically;
    these numerical masses are explicit approximation, not anatomical tissue.
    The authored source is never changed and the generated asset is cached.
    """
    from pxr import Usd, UsdGeom, UsdPhysics, Gf
    if not math.isfinite(frame_mass) or frame_mass<=0:
        raise ValueError('frame_mass must be positive and finite')
    source=Path(usd_path).read_text()
    digest=hashlib.sha256((source+str(frame_mass)+Path(__file__).read_text()).encode()).hexdigest()
    root=Path(os.environ.get('PROTOMOTIONS_HUMAN_MODEL_CACHE',str(Path(tempfile.gettempdir())/f'protomotions-human-model-{os.getuid()}')))/digest[:24]
    root.mkdir(parents=True,exist_ok=True)
    destination=root/'smpl_humanoid_serial.usda'
    if destination.exists():return destination
    stage=Usd.Stage.CreateInMemory();stage.GetRootLayer().ImportFromString(source)
    joints=[prim for prim in stage.Traverse() if prim.GetTypeName()=='PhysicsJoint']
    if len(joints)!=23:raise ValueError('Expected 23 three-axis SMPL joints')
    for prim in joints:
        joint=UsdPhysics.Joint(prim);name=prim.GetName();path=prim.GetPath()
        parent=joint.GetBody0Rel().GetTargets()[0];child=joint.GetBody1Rel().GetTargets()[0]
        pos0=joint.GetLocalPos0Attr().Get();pos1=joint.GetLocalPos1Attr().Get()
        rot0=joint.GetLocalRot0Attr().Get();rot1=joint.GetLocalRot1Attr().Get()
        if rot0!=Gf.Quatf(1) or rot1!=Gf.Quatf(1) or pos1!=Gf.Vec3f(0):
            raise ValueError(f'Unsupported nonidentity SMPL joint frame: {name}')
        bounds=[(prim.GetAttribute(f'limit:rot{axis}:physics:low').Get(),prim.GetAttribute(f'limit:rot{axis}:physics:high').Get()) for axis in 'XYZ']
        transform=stage.GetPrimAtPath(child).GetAttribute('xformOp:transform').Get()
        frames=[]
        for axis in 'xy':
            frame_path=child.GetParentPath().AppendChild(f'_joint_frame_{name}_{axis}')
            frame=UsdGeom.Xform.Define(stage,frame_path)
            frame.AddTransformOp().Set(transform)
            UsdPhysics.RigidBodyAPI.Apply(frame.GetPrim())
            mass=UsdPhysics.MassAPI.Apply(frame.GetPrim())
            mass.CreateMassAttr(frame_mass);mass.CreateCenterOfMassAttr(Gf.Vec3f(0))
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(frame_mass*1e-3));mass.CreatePrincipalAxesAttr(Gf.Quatf(1))
            frames.append(frame_path)
        stage.RemovePrim(path)
        chain=[parent,*frames,child]
        for i,axis in enumerate('XYZ'):
            hinge=UsdPhysics.RevoluteJoint.Define(stage,path.GetParentPath().AppendChild(f'{name}_{axis.lower()}'))
            hinge.CreateAxisAttr(axis);hinge.CreateBody0Rel().SetTargets([chain[i]])
            hinge.CreateBody1Rel().SetTargets([chain[i+1]])
            hinge.CreateLocalPos0Attr(pos0 if i==0 else Gf.Vec3f(0));hinge.CreateLocalPos1Attr(Gf.Vec3f(0))
            hinge.CreateLocalRot0Attr(Gf.Quatf(1));hinge.CreateLocalRot1Attr(Gf.Quatf(1))
            hinge.CreateLowerLimitAttr(bounds[i][0]);hinge.CreateUpperLimitAttr(bounds[i][1])
            drive=UsdPhysics.DriveAPI.Apply(hinge.GetPrim(),'angular')
            drive.CreateStiffnessAttr(0);drive.CreateDampingAttr(0);drive.CreateTypeAttr('force')
    _atomic_write(destination,stage.GetRootLayer().ExportToString())
    return destination
