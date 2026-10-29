# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""Physiological sensory (afferent) delay for actor observations (candidate, 2026-10-28).

Plant / sensor side: the environment keeps a ring buffer of past observations at the control rate and gives the
actor a delayed copy of the body-state observations under ``<key_prefix><key>``. The original keys stay undelayed,
so the critic, the AMP discriminator and its critic keep seeing the true state (asymmetric actor-critic).

Delay per env and per channel group is drawn at reset: proprio (+ cutaneous / contact) d ~ U(proprio_ms), vestibular
d ~ U(vestib_ms); ``fixed_*_ms`` replaces the draw (evaluation). A delay between two control steps is a linear
interpolation of the two bracketing buffer slots (55 ms at 30 Hz = mix of t - 33 ms and t - 67 ms). On reset an env's
buffer is filled with its first observation. Command / goal inputs, the gait clock, previous actions and terrain are
never delayed (they are not listed in a group). Default ``enabled=False``: nothing is created and the env is unchanged.
"""
from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class SensoryDelayConfig:
    enabled: bool = False
    # Group "proprio": joint / body state and contact (proprioceptive + cutaneous afferents).
    proprio_keys: List[str] = field(default_factory=lambda: ["max_coords_obs", "historical_max_coords_obs"])
    proprio_ms: Tuple[float, float] = (40.0, 70.0)
    # Group "vestib": pelvis gravity direction and head specific force (vestibular afferents).
    vestib_keys: List[str] = field(default_factory=lambda: ["se_gravity", "se_specific_force"])
    vestib_ms: Tuple[float, float] = (30.0, 60.0)
    # Evaluation: a fixed delay instead of the per-env draw (None = draw from the range).
    fixed_proprio_ms: Optional[float] = None
    fixed_vestib_ms: Optional[float] = None
    max_delay_ms: float = 100.0        # buffer depth = floor(max / dt) + 2 slots
    key_prefix: str = "delayed_"

    def groups(self) -> Dict[str, Tuple[List[str], Tuple[float, float], Optional[float]]]:
        return {"proprio": (list(self.proprio_keys), tuple(self.proprio_ms), self.fixed_proprio_ms),
                "vestib": (list(self.vestib_keys), tuple(self.vestib_ms), self.fixed_vestib_ms)}

    def delayed_keys(self) -> List[str]:
        return [k for keys, _, _ in self.groups().values() for k in keys]


class SensoryDelayBuffer:
    """Vectorized ring buffer [depth, num_envs, dim] per delayed key, shared write pointer, per-env read delay."""

    def __init__(self, config: SensoryDelayConfig, num_envs: int, dt: float, device, generator=None):
        self.config = config
        self.num_envs = num_envs
        self.dt_ms = 1000.0 * float(dt)
        self.device = torch.device(device)
        self.generator = generator
        groups = config.groups()
        self.group_names = list(groups)
        seen = set()
        for name, (keys, (lo, hi), fixed) in groups.items():
            if lo > hi or lo < 0:
                raise ValueError(f"sensory delay {name}: bad range {lo, hi}")
            top = hi if fixed is None else fixed
            if top > config.max_delay_ms or (fixed is not None and fixed < 0):
                raise ValueError(f"sensory delay {name}: {top} ms outside [0, max_delay_ms={config.max_delay_ms}]")
            for k in keys:
                if k in seen:
                    raise ValueError(f"sensory delay: key {k} in two groups")
                seen.add(k)
        self.key_group = {k: i for i, (keys, _, _) in enumerate(groups.values()) for k in keys}
        self.depth = int(math.floor(config.max_delay_ms / self.dt_ms + 1e-6)) + 2
        self.head = 0                                        # slot of the newest (age 0) observation
        self.buffers: Dict[str, Tensor] = {}
        self.delay_ms = torch.zeros(num_envs, len(groups), device=self.device)   # per env, per group (logging)
        self._env_index = torch.arange(num_envs, device=self.device)

    def _sample_delays(self, env_ids: Tensor):
        for g, (_, (lo, hi), fixed) in enumerate(self.config.groups().values()):
            if fixed is not None:
                self.delay_ms[env_ids, g] = float(fixed)
            else:
                u = torch.rand(len(env_ids), device=self.device, generator=self.generator)
                self.delay_ms[env_ids, g] = lo + (hi - lo) * u

    def reset(self, env_ids: Tensor, obs: Dict[str, Tensor]):
        """Fill every slot of env_ids with their current observation and draw new delays."""
        env_ids = env_ids.to(self.device, torch.long)
        for k in self.key_group:
            if k not in obs:
                raise KeyError(f"sensory delay: observation {k} is not an env observation component")
            x = obs[k]
            if k not in self.buffers:
                self.buffers[k] = torch.zeros(self.depth, *x.shape, dtype=x.dtype, device=self.device)
            self.buffers[k][:, env_ids] = x[env_ids].unsqueeze(0)
        self._sample_delays(env_ids)

    def push(self, obs: Dict[str, Tensor]):
        """Append the observation of the control step that just ended (all envs)."""
        self.head = (self.head + 1) % self.depth
        for k, buf in self.buffers.items():
            buf[self.head].copy_(obs[k])

    def read(self) -> Dict[str, Tensor]:
        """Delayed observations {key_prefix + key: [num_envs, dim]}."""
        age = self.delay_ms / self.dt_ms                      # [num_envs, groups], in control steps
        i0 = torch.floor(age).long().clamp(0, self.depth - 1)
        frac = (age - i0.to(age.dtype)).clamp(0.0, 1.0)
        i1 = (i0 + 1).clamp(max=self.depth - 1)
        s0 = (self.head - i0) % self.depth
        s1 = (self.head - i1) % self.depth
        out = {}
        for k, buf in self.buffers.items():
            g = self.key_group[k]
            a = buf[s0[:, g], self._env_index]
            b = buf[s1[:, g], self._env_index]
            f = frac[:, g].unsqueeze(-1).to(buf.dtype)
            out[self.config.key_prefix + k] = a + f * (b - a)
        return out

    def log_values(self) -> Dict[str, Tensor]:
        return {f"sensory_delay_{n}_ms": self.delay_ms[:, i] for i, n in enumerate(self.group_names)}

    def save_state(self) -> dict:
        return {"head": self.head, "delay_ms": self.delay_ms.clone(),
                "buffers": {k: v.clone() for k, v in self.buffers.items()}}

    def load_state(self, state: dict):
        self.head = state["head"]
        self.delay_ms.copy_(state["delay_ms"])
        for k, v in state["buffers"].items():
            self.buffers[k].copy_(v)


def delay_actor_inputs(agent_cfg, delay_cfg: SensoryDelayConfig):
    """Point the actor (and its mu model) at the delayed keys; critic, discriminator and disc critic stay undelayed.

    Only the key names change (same order and sizes), so the actor weights and its input normalizer load unchanged.
    """
    if delay_cfg is None or not delay_cfg.enabled:
        return agent_cfg
    keys = set(delay_cfg.delayed_keys())
    m = agent_cfg.model
    for module in (m.actor, m.actor.mu_model):
        module.in_keys = [delay_cfg.key_prefix + k if k in keys else k for k in module.in_keys]
    m.in_keys = list(m.in_keys) + [k for k in m.actor.in_keys if k not in m.in_keys]
    return agent_cfg
