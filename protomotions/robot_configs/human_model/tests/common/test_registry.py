# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Migration contracts: aliases, resources, ordering and serialized imports."""
import hashlib
import importlib
import json
import pickle
from pathlib import Path

from protomotions.robot_configs.human_model.registry import get_model
from protomotions.robot_configs.human_model.profile import load_profile
from protomotions.robot_configs.smpl import SmplRobotConfig


def test_legacy_imports_resolve_to_canonical_module_objects():
    for name in ("profile", "dynamics", "integration", "metrics", "validation", "calibration"):
        old = importlib.import_module(f"protomotions.robot_configs.human_model.{name}")
        new = importlib.import_module(f"protomotions.robot_configs.human_model.common.{name}")
        assert old is new
    old_class = pickle.loads(b"cprotomotions.robot_configs.human_model.dynamics\nHumanJointModel\n.")
    from protomotions.robot_configs.human_model.common.dynamics import HumanJointModel
    assert old_class is HumanJointModel


def test_legacy_profile_alias_and_canonical_model_select_same_values(monkeypatch):
    monkeypatch.delenv("PROTOMOTIONS_HUMAN_MODEL", raising=False)
    assert load_profile() == load_profile("human_model_v1")
    assert get_model("healthy_adult_v1") == get_model("human_model_v1")
    old = SmplRobotConfig()
    new = SmplRobotConfig(human_model_profile="human_model_v1")
    assert old.kinematic_info.dof_names == new.kinematic_info.dof_names
    assert old.human_model_parameters == new.human_model_parameters == {}


def test_v1_joint_map_follows_actual_robot_order():
    model = get_model()
    mapping = json.loads((model.root / "joint_map.json").read_text())
    assert [r["name"] for r in mapping["joints"]] == list(SmplRobotConfig().kinematic_info.dof_names)
    assert len(mapping["joints"]) == 69


def test_frozen_resources_and_candidate_profile_are_not_promoted():
    model = get_model()
    manifest = json.loads((model.root / "manifest.json").read_text())
    for row in manifest["artifacts"]:
        path = model.root / row["path"]
        assert path.is_file(), path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], path
    assert manifest["default_profile"] == "profiles/healthy_adult_v1.json"
    assert manifest["default_features"] == []
    assert manifest["population_profile"]["status"] == "candidate"
    root = model.root.parent
    for name in ("healthy_adult_v1.json", "candidate_parameters.json", "validation_contract.json"):
        assert (root / name).resolve() == model.root / "profiles" / name
    old_usd = root.parents[1] / "data/assets/usd/smpl_humanoid_healthy_adult_v1.usda"
    assert old_usd.resolve() == model.asset_path("usd")
