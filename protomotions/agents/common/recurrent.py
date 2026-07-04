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
"""Recurrent (GRU) actor core — a drop-in replacement for MLPWithConcat.

Motivation: model the ④ MotionDriver as a recurrent dynamical system, matching the
Churchland/Shenoy view of primary motor cortex (M1) as a dynamical system whose
population state evolves over time to generate movement. ProtoMotions' agents are
feedforward (the policy is called per timestep, the optimizer samples individual
timesteps), so rather than thread a hidden state across environment steps — which
would require agent surgery — this core unrolls a GRU over the recent motor-state
history that is ALREADY present in the observation (`historical_max_coords_obs`, a
dilated window of past states). Each timestep the GRU processes that short sequence
into a hidden state, which is combined with the current observation to emit the
action. This is a self-contained recurrent core; no changes to the rollout/optimizer.

Interface matches MLPWithConcat (TensorDict in/out, same in_keys/out_keys), so it is
selected purely by pointing an actor's `mu_model._target_` at this class.
"""
from typing import List

import torch
from torch import nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase

from protomotions.agents.common.common import NormObsBase
from protomotions.agents.common.mlp import build_mlp
from protomotions.agents.utils.training import get_activation_func
from protomotions.agents.common.config import RecurrentMLPWithConcatConfig


class RecurrentMLPWithConcat(TensorDictModuleBase):
    """GRU-over-history actor core. Normalizes the concatenated observation exactly like
    MLPWithConcat, reshapes the `history_key` slice into a [B, T, F] sequence, runs a GRU
    over it (chronological order), then feeds [other normalized obs, final GRU hidden] to
    an MLP head. GRU input size and all Linear sizes are inferred on the first forward."""

    config: RecurrentMLPWithConcatConfig

    def __init__(self, config: RecurrentMLPWithConcatConfig):
        TensorDictModuleBase.__init__(self)
        self.config = config
        assert config.in_keys, "RecurrentMLPWithConcat requires in_keys."
        assert config.out_keys and len(config.out_keys) == 1, "requires exactly one out_key."
        assert config.history_key in config.in_keys, "history_key must be one of in_keys."

        self.norm = NormObsBase(config)
        self.head = build_mlp(config)  # hidden layers + final LazyLinear(num_out)
        self.gru: nn.GRU = None  # lazily built on first forward (needs per-step feature dim)
        self.output_activation = (
            get_activation_func(config.output_activation) if config.output_activation else None
        )

        self.in_keys: List[str] = list(config.in_keys)
        self.out_keys: List[str] = list(config.out_keys)

    def _ensure_gru(self, feat_dim: int, device):
        if self.gru is None:
            self.gru = nn.GRU(
                input_size=feat_dim,
                hidden_size=self.config.gru_hidden_size,
                num_layers=self.config.gru_num_layers,
                batch_first=True,
            ).to(device)

    def forward(self, tensordict: TensorDict) -> TensorDict:
        parts = [tensordict[k] for k in self.config.in_keys]
        combined = torch.cat(parts, dim=-1)
        norm = self.norm(combined)  # [B, sum(dims)]; per-dim running normalization

        # Split the normalized vector back into its per-key pieces (per-dim norm commutes
        # with slicing) so the history piece keeps its sequence structure.
        sizes = [p.shape[-1] for p in parts]
        pieces = dict(zip(self.config.in_keys, torch.split(norm, sizes, dim=-1)))

        hist = pieces[self.config.history_key]  # [B, T*F]
        T = self.config.num_history_steps
        B = hist.shape[0]
        F = hist.shape[-1] // T
        seq = hist.view(B, T, F).flip(1)  # flip: history is most-recent-first → feed oldest→newest

        self._ensure_gru(F, hist.device)
        _, h_n = self.gru(seq)  # h_n: [num_layers, B, H]
        h = h_n[-1]  # [B, H] — final-layer hidden state after the newest step

        others = [pieces[k] for k in self.config.in_keys if k != self.config.history_key]
        feat = torch.cat(others + [h], dim=-1)
        out = self.head(feat)
        if self.output_activation is not None:
            out = self.output_activation(out)

        tensordict[self.config.out_keys[0]] = out
        return tensordict
