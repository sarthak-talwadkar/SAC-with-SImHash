"""A fixed-size replay buffer (off-policy memory) backed by NumPy arrays.

SAC is off-policy: it learns from a big pool of past transitions, reusing each
many times. We store transitions as a ring buffer and sample uniform batches.
"""
from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, size: int, device):
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.act = np.zeros((size, act_dim), dtype=np.float32)
        self.rew = np.zeros(size, dtype=np.float32)
        self.done = np.zeros(size, dtype=np.float32)  # 1.0 only for TRUE terminations
        self.max_size = size
        self.ptr = 0
        self.size = 0
        self.device = device

    def add(self, obs, act, rew, next_obs, done) -> None:
        i = self.ptr
        self.obs[i] = obs
        self.act[i] = act
        self.rew[i] = rew
        self.next_obs[i] = next_obs
        self.done[i] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> dict:
        idx = np.random.randint(0, self.size, size=batch_size)
        to_t = lambda a: torch.as_tensor(a, device=self.device)
        return dict(
            obs=to_t(self.obs[idx]),
            act=to_t(self.act[idx]),
            rew=to_t(self.rew[idx]),
            next_obs=to_t(self.next_obs[idx]),
            done=to_t(self.done[idx]),
        )

    def __len__(self) -> int:
        return self.size

    def state_dict(self) -> dict:
        """Serialize only the filled portion of the buffer (for resuming)."""
        n = self.size
        return dict(
            obs=self.obs[:n].copy(),
            next_obs=self.next_obs[:n].copy(),
            act=self.act[:n].copy(),
            rew=self.rew[:n].copy(),
            done=self.done[:n].copy(),
            ptr=self.ptr,
            size=self.size,
        )

    def load_state_dict(self, sd: dict) -> None:
        n = int(sd["size"])
        if n > self.max_size:
            raise ValueError(f"saved buffer ({n}) exceeds current capacity ({self.max_size})")
        self.obs[:n] = sd["obs"]
        self.next_obs[:n] = sd["next_obs"]
        self.act[:n] = sd["act"]
        self.rew[:n] = sd["rew"]
        self.done[:n] = sd["done"]
        self.ptr = int(sd["ptr"])
        self.size = n
