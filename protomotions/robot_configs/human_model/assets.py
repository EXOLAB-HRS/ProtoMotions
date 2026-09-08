# SPDX-License-Identifier: Apache-2.0
"""Derive joint-only MJCF/USD assets locally; never rewrite upstream geometry."""

import hashlib
import json
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


def render_usda(source, profile, backend_limits):
    if source.startswith("version https://git-lfs.github.com/spec"):
        raise ValueError("SMPL USD is a Git LFS pointer. Run git lfs pull in ProtoMotions before using IsaacLab.")
    if not source.startswith("#usda"):
        raise ValueError("Expected the text SMPL USDA asset; binary USD needs explicit conversion")
    seen = set()

    def replace_joint(match):
        block, body = match.group(0), match.group(1)
        for axis in "xyz":
            name = f"{body}_{axis}"
            row = profile["joints"].get(name)
            if row is None:
                raise ValueError(f"Unexpected USD joint: {name}")
            token = f"rot{axis.upper()}"
            if f'custom token mjcf:{token}:name = "{name}"' not in block:
                raise ValueError(f"USD joint/axis mapping changed: {name}")
            props = {
                f"limit:{token}:physics:low": row["rom_deg"][0],
                f"limit:{token}:physics:high": row["rom_deg"][1],
                f"drive:{token}:physics:stiffness": 0,
                f"drive:{token}:physics:damping": 0,
                f"drive:{token}:physics:maxForce": backend_limits[name],
                # Disable legacy soft-limit spring gains, retaining hard limits.
                f"physxLimit:{token}:stiffness": 0,
                f"physxLimit:{token}:damping": 0,
            }
            for prop, value in props.items():
                block, count = re.subn(
                    rf"(?m)^(\s*float {re.escape(prop)} = )[^\n]+$",
                    lambda m: m.group(1) + format(value, ".12g"), block,
                )
                if count != 1:
                    raise ValueError(f"Expected exactly one USD property {body}/{prop}")
            seen.add(name)
        return block

    result = re.sub(r'        def PhysicsJoint "([^"]+)" \([\s\S]*?\n        }', replace_joint, source)
    if seen != set(profile["joints"]):
        raise ValueError("USD/profile joint mismatch")
    return result.replace("#usda 1.0", "#usda 1.0\n# Locally derived human joint properties; see robot_configs/human_model/README.md", 1)


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


def derive_assets(xml_path, usd_path, profile, backend_limits, require_usd=False):
    xml_path = Path(xml_path)
    xml = xml_path.read_text()
    usd = Path(usd_path).read_text() if require_usd else None
    digest = hashlib.sha256()
    for data in (xml, usd or "", json.dumps(profile, sort_keys=True), json.dumps(backend_limits, sort_keys=True), Path(__file__).read_text()):
        digest.update(data.encode())
    root = Path(os.environ.get("PROTOMOTIONS_HUMAN_MODEL_CACHE", str(Path(tempfile.gettempdir()) / f"protomotions-human-model-{os.getuid()}")))
    destination = root / digest.hexdigest()[:24]
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    derived_xml = destination / "smpl_humanoid.xml"
    if not derived_xml.exists():
        _atomic_write(derived_xml, render_mjcf(xml, profile, backend_limits))
    if require_usd:
        derived_usd = destination / "smpl_humanoid.usda"
        if not derived_usd.exists():
            _atomic_write(derived_usd, render_usda(usd, profile, backend_limits))
    return destination
