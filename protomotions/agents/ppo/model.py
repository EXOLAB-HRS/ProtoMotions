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
"""PPO model implementation with actor-critic architecture.

This module implements the neural network models for Proximal Policy Optimization.
The actor outputs a Gaussian policy distribution, and the critic estimates state values.

Key Classes:
    - PPOActor: Policy network with Gaussian action distribution
    - PPOModel: Complete actor-critic model for PPO
"""

import torch
from torch import distributions, nn
from protomotions.utils.hydra_replacement import get_class
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase
from protomotions.agents.common.common import ModuleContainer
from protomotions.agents.ppo.config import PPOActorConfig, PPOModelConfig
from protomotions.agents.base_agent.model import BaseModel


def _hc_residual(actor, tensordict, mu):
    """Residual teacher (2026-11-12, human-controller arm fix on teacher 344, off unless HC_RESID_BASE is set).
    mu_out = mu_base + w * (mu - mu_base): mu_base is a frozen copy of the actor mean loaded from HC_RESID_BASE (the
    parent checkpoint), mu the trainable actor (initialised from the same parent, so the residual starts at 0).
    w = 1 on HC_RESID_IDX (comma list of action indices, e.g. the arm joints), HC_RESID_W elsewhere (default 0: those
    outputs stay exactly the parent's). The frozen copy is kept out of the module tree (not in state_dict / optimiser)."""
    import os
    path = os.environ.get("HC_RESID_BASE")
    if not path:
        return mu
    st = actor.__dict__.get("_hc_resid")
    if st is None:
        import copy
        sd = torch.load(path, map_location="cpu", weights_only=False)["model"]
        base = copy.deepcopy(actor.mu)
        base.load_state_dict({k[len("_actor.mu."):]: v for k, v in sd.items() if k.startswith("_actor.mu.")})
        base = base.to(mu.device).eval()
        for p_ in base.parameters():
            p_.requires_grad_(False)
        w = torch.full((mu.shape[-1],), float(os.environ.get("HC_RESID_W", "0")), device=mu.device)
        idx = [int(i) for i in os.environ.get("HC_RESID_IDX", "").split(",") if i.strip()]
        w[idx] = 1.0
        w[[int(i) for i in os.environ.get("HC_RESID_ZERO_IDX", "").split(",") if i.strip()]] = 0.0   # 2026-11-13: locked outputs
        st = actor.__dict__["_hc_resid"] = dict(base=base, w=w)
        print(f"[hc_resid] base {path} trainable idx {idx} other w {float(os.environ.get('HC_RESID_W', '0'))}", flush=True)
    with torch.no_grad():
        mb = st["base"](tensordict.clone(False))[actor.config.mu_key]
    return mb + st["w"] * (mu - mb)


class PPOActor(TensorDictModuleBase):
    """PPO policy network (actor).

    Self-contained policy that computes distribution parameters, samples actions,
    and computes log probabilities all in a single forward pass.

    Args:
        config: Actor configuration including network architecture and initial log std.

    Attributes:
        logstd: Log standard deviation parameter (typically fixed during training).
        mu: Neural network that outputs action means.
        in_keys: List of input keys from mu model.
        out_keys: List of output keys (action, mean_action, neglogp).
    """

    def __init__(self, config: PPOActorConfig):
        super().__init__()
        self.config = config
        logstd = torch.ones(self.config.num_out) * self.config.actor_logstd
        for start, end, value in getattr(self.config, "logstd_slices", None) or []:
            logstd[int(start):int(end)] = float(value)
        self.logstd = nn.Parameter(logstd, requires_grad=self.config.learnable_std)
        MuClass = get_class(self.config.mu_model._target_)
        self.mu: TensorDictModuleBase = MuClass(config=self.config.mu_model)

        self.in_keys = self.config.in_keys
        self.out_keys = self.config.out_keys
        for key in ["action", "mean_action", "neglogp"]:
            assert (
                key in self.out_keys
            ), f"PPOActor output key {key} not in out_keys {self.out_keys}"

    def forward(self, tensordict: TensorDict) -> TensorDict:
        """Forward pass: compute mu/std, sample action, compute neglogp.

        Args:
            tensordict: TensorDict containing observations.

        Returns:
            TensorDict with action, mean_action, and neglogp added.
        """
        # Compute distribution parameters
        tensordict = self.mu(tensordict)
        mu = tensordict[self.config.mu_key]
        mu = _hc_residual(self, tensordict, mu)
        tensordict[self.config.mu_key] = mu
        std = torch.exp(self.logstd)

        # Sample action from distribution
        # mu * 0 + std broadcasts std to match mu's batch shape while preserving
        # the computational graph when std is learnable
        dist = distributions.Normal(mu, mu * 0 + std)
        action = dist.sample()

        # Compute negative log probability
        neglogp = -dist.log_prob(action).sum(dim=-1)

        # Store all outputs
        tensordict["action"] = action
        tensordict["mean_action"] = mu
        tensordict["neglogp"] = neglogp

        return tensordict


class PPOModel(BaseModel):
    """Complete PPO model with actor and critic networks.

    Pure forward function that computes all model outputs in TensorDict.
    The forward pass adds action distribution parameters and value estimates.

    Args:
        config: Model configuration specifying actor and critic architectures.

    Attributes:
        _actor: Policy network.
        _critic: Value network.
    """

    config: PPOModelConfig

    def __init__(self, config: PPOModelConfig):
        super().__init__(config)

        # create networks
        ActorClass = get_class(self.config.actor._target_)
        self._actor: PPOActor = ActorClass(config=self.config.actor)

        CriticClass = get_class(self.config.critic._target_)
        self._critic: ModuleContainer = CriticClass(config=self.config.critic)

        # Set in_keys from actor (actor inherits from mu model)
        actor_critic_in_keys = list(set(self._actor.in_keys + self._critic.in_keys))
        actor_critic_out_keys = list(set(self._actor.out_keys + self._critic.out_keys))
        for key in actor_critic_out_keys:
            assert (
                key in self.config.out_keys
            ), f"PPOModel output key {key} not in out_keys {self.config.out_keys}"
        for key in actor_critic_in_keys:
            assert (
                key in self.config.in_keys
            ), f"PPOModel input key {key} not in in_keys {self.config.in_keys}"

        self.in_keys = self.config.in_keys
        self.out_keys = self.config.out_keys

    def forward(self, tensordict: TensorDict) -> TensorDict:
        """Forward pass through actor and critic.

        Computes all outputs:
        - action: Sampled action
        - mean_action: Deterministic action (mean)
        - neglogp: Negative log probability of sampled action
        - value: State value estimate

        Args:
            tensordict: TensorDict containing observations.

        Returns:
            TensorDict with all model outputs added.
        """
        # Actor forward: adds action, mean_action, neglogp
        tensordict = self._actor(tensordict)

        # Critic forward: adds value estimate
        tensordict = self._critic(tensordict)

        return tensordict
