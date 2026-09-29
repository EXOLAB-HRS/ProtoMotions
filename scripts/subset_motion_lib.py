#!/usr/bin/env python3
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

"""
Script to create a subset of a motion library by sampling every N motions.
Might be useful if you realize your GPU cannot load large motion libraries.
"""

import torch
import copy
import hashlib
from pathlib import Path


def subset_motion_lib(input_path: str, output_path: str, sample_every: int = 200, indices=None,
                      weights=None):
    """
    Load a motion library and create a subset by sampling every N motions.
    
    Args:
        input_path: Path to input .pt motion library
        output_path: Path to output .pt file
        sample_every: Take every Nth motion (default: 200)
        indices: Explicit motion indices to keep. Overrides sample_every, which
            cannot express a content-based selection such as "locomotion classes
            that match the task being trained".
        weights: Sampling weight per selected motion, in the selection order.
            Motion weights drive both reference-state initialisation and the
            expert batches an adversarial discriminator is trained against, so
            they are how a library is matched to the command range being trained.
            Stored normalised; the source weights are kept when omitted.
    """
    print(f"Loading motion library from {input_path}")
    data = torch.load(input_path, map_location="cpu", weights_only=False)
    
    # Get number of motions
    num_motions = len(data["motion_lengths"])
    print(f"Original motion library has {num_motions} motions")
    
    # Select motion indices
    if indices is not None:
        selected_indices = [int(i) for i in indices]
        out_of_range = [i for i in selected_indices if not 0 <= i < num_motions]
        if out_of_range:
            raise ValueError(f"motion indices outside the library: {out_of_range}")
        print(f"Selecting {len(selected_indices)} motions by explicit index")
    else:
        selected_indices = list(range(0, num_motions, sample_every))
        print(f"Selecting {len(selected_indices)} motions (every {sample_every}th)")
    num_selected = len(selected_indices)
    if not selected_indices or len(set(selected_indices)) != num_selected:
        raise ValueError("Select at least one motion without duplicate indices")
    
    # Get the frame ranges for each selected motion
    length_starts = data["length_starts"]
    motion_num_frames = data["motion_num_frames"]
    
    # Collect frame indices for selected motions
    frame_indices = []
    new_motion_num_frames = []
    new_motion_lengths = []
    new_motion_dt = []
    new_motion_weights = []
    new_motion_files = []
    
    for idx in selected_indices:
        start = length_starts[idx].item()
        num_frames = motion_num_frames[idx].item()
        frame_indices.extend(range(start, start + num_frames))
        new_motion_num_frames.append(num_frames)
        new_motion_lengths.append(data["motion_lengths"][idx].item())
        new_motion_dt.append(data["motion_dt"][idx].item())
        new_motion_weights.append(data["motion_weights"][idx].item())
        if "motion_files" in data:
            new_motion_files.append(data["motion_files"][idx])
    
    frame_indices = torch.tensor(frame_indices, dtype=torch.long)
    
    # Create new data dictionary
    new_data = {}
    
    # Tensor fields that are indexed by frame
    frame_indexed_fields = ["gts", "grs", "gvs", "gavs", "dvs", "dps", "contacts"]
    if "lrs" in data and data["lrs"] is not None:
        frame_indexed_fields.append("lrs")
    
    for field in frame_indexed_fields:
        if field in data and data[field] is not None:
            new_data[field] = data[field][frame_indices]
            print(f"  {field}: {data[field].shape} -> {new_data[field].shape}")
    
    # Rebuild length_starts
    new_motion_num_frames_tensor = torch.tensor(new_motion_num_frames, dtype=torch.long)
    lengths_shifted = new_motion_num_frames_tensor.roll(1)
    lengths_shifted[0] = 0
    new_data["length_starts"] = lengths_shifted.cumsum(0)
    
    # Other motion-indexed fields
    new_data["motion_num_frames"] = new_motion_num_frames_tensor
    new_data["motion_lengths"] = torch.tensor(new_motion_lengths, dtype=torch.float32)
    new_data["motion_dt"] = torch.tensor(new_motion_dt, dtype=torch.float32)
    if weights is None:
        new_data["motion_weights"] = torch.tensor(new_motion_weights, dtype=torch.float32)
    else:
        requested = torch.tensor([float(w) for w in weights], dtype=torch.float32)
        if len(requested) != num_selected:
            raise ValueError(f"got {len(requested)} weights for {num_selected} motions")
        if not torch.isfinite(requested).all() or (requested < 0).any() or requested.sum() <= 0:
            raise ValueError("weights must be finite, non-negative and not all zero")
        new_data["motion_weights"] = requested / requested.sum()
    
    if new_motion_files:
        new_data["motion_files"] = tuple(new_motion_files)

    # Keep the model contract and clip-level screening evidence for v2 loaders.
    # Selection never upgrades candidate references to training-approved data.
    if "human_model_metadata" in data:
        metadata = copy.deepcopy(data["human_model_metadata"])
        if "motions" in metadata:
            if len(metadata["motions"]) != num_motions:
                raise ValueError("Reference metadata does not match the motion count")
            metadata["motions"] = [metadata["motions"][i] for i in selected_indices]
        metadata["subset_parent"] = {
            "path": str(input_path),
            "sha256": hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
            "indices": selected_indices,
            "weights_overridden": weights is not None,
        }
        new_data["human_model_metadata"] = metadata
    
    # Save
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_data, output_path)
    
    print(f"\nSaved subset to {output_path}")
    print(f"  Motions: {num_motions} -> {num_selected}")
    print(f"  Total frames: {len(data['gts'])} -> {len(new_data['gts'])}")


if __name__ == "__main__":
    import argparse

    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--input", required=True)
    cli.add_argument("--output", required=True)
    cli.add_argument("--sample-every", type=int, default=200)
    cli.add_argument("--indices", type=int, nargs="+", default=None)
    cli.add_argument("--weights", type=float, nargs="+", default=None)
    parsed = cli.parse_args()
    subset_motion_lib(parsed.input, parsed.output, parsed.sample_every, parsed.indices,
                      parsed.weights)
