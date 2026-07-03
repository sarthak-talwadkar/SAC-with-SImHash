"""SimHash count-based exploration bonus (Tang et al., 2017, "#Exploration").

Idea: hash each continuous state to a short binary code via a FIXED random
projection + sign (locality-sensitive hashing), so similar states share a code.
Count how often each code is visited and reward rare codes:

    code(s) = sign(A · s) in {0,1}^k          (A fixed, ~ N(0,1))
    bonus   = beta / sqrt( count[ code(s') ] )

The projection A never changes during a run, so a state's code is stable and
the counts stay meaningful.

Variants (this project's ablation of the hash geometry):

  raw    : sgn(A·s)        -- the published formulation. Sign hashing is
           scale-invariant (sgn(A·λs) = sgn(A·s) for λ>0): codes are constant
           on rays through the origin, so k central hyperplanes carve R^d into
           at most 2*sum_{i<d} C(k-1,i) regions -- only 2k sectors when d=2 --
           and magnitude information is lost entirely. The N(0,1) projection
           also weights all dimensions equally, so dimensions with small
           numeric range (e.g. MountainCar velocity) barely influence the code.
  norm   : sgn(A·s~)       -- standardize obs by a running mean/std first, so
           every dimension influences the code equally.
  bias   : sgn(A·[s; 1])   -- append a constant, making the hyperplanes AFFINE:
           they no longer pass through the origin, so codes distinguish
           magnitude and the sector cap no longer applies.
  affine : sgn(A·[s~; 1])  -- both fixes (the proposed corrected hash).

Caveat for `norm`/`affine`: while the running statistics are still moving, the
same state can map to different codes at different times, mildly polluting the
counts. Stats effectively settle during the random-action warmup, and the count
table is large, so in practice this is noise on the first few thousand steps.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from .base import ExplorationBonus
from utils import RunningMeanStd


class SimHashBonus(ExplorationBonus):
    name = "simhash"

    def __init__(self, obs_dim: int, cfg, device, seed: int = 0):
        self.k = int(cfg.k)
        self.beta = float(cfg.beta)
        self.normalize = bool(getattr(cfg, "normalize", False))
        self.bias = bool(getattr(cfg, "bias", False))
        self.device = device
        # Fixed random projection A: (k, D). Seeded for reproducibility.
        # With the bias variant the projection sees one extra constant input.
        in_dim = obs_dim + (1 if self.bias else 0)
        rng = np.random.default_rng(seed + 0xA5)
        self.A = rng.standard_normal((self.k, in_dim)).astype(np.float32)
        self.counts: dict[bytes, int] = defaultdict(int)
        self.obs_rms = RunningMeanStd(shape=(obs_dim,)) if self.normalize else None

    def _features(self, states: np.ndarray) -> np.ndarray:
        """Apply the variant's preprocessing: normalization and/or bias input."""
        x = states
        if self.normalize:
            x = (x - self.obs_rms.mean) / self.obs_rms.std
            x = np.clip(x, -5.0, 5.0).astype(np.float32)
        if self.bias:
            x = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)
        return x

    def _keys(self, states: np.ndarray) -> list[bytes]:
        """Map a [B, obs_dim] array to a list of B hashable byte-codes."""
        proj = self._features(states) @ self.A.T   # [B, k]
        bits = proj > 0.0                          # [B, k] boolean sign pattern
        packed = np.packbits(bits, axis=1)         # compact bytes, works for any k
        return [row.tobytes() for row in packed]

    def observe(self, next_obs) -> None:
        arr = np.asarray(next_obs, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        if self.normalize:
            self.obs_rms.update(arr)  # track stats from true visitation
        for key in self._keys(arr):
            self.counts[key] += 1

    @torch.no_grad()
    def compute_bonus(self, obs, act, next_obs) -> torch.Tensor:
        arr = next_obs.detach().cpu().numpy().astype(np.float32)
        keys = self._keys(arr)
        # Sampled states were counted when first visited, so count >= 1; the
        # default of 1 is just a safety net against a never-seen code.
        counts = np.fromiter((self.counts.get(k, 1) for k in keys),
                             dtype=np.float32, count=len(keys))
        bonus = self.beta / np.sqrt(counts)
        return torch.as_tensor(bonus, dtype=torch.float32, device=self.device)

    def diagnostics(self) -> dict:
        return dict(simhash_unique_codes=float(len(self.counts)))

    def state_dict(self) -> dict:
        # The count table IS the learned state; A is fixed but saved for safety.
        sd = dict(counts=dict(self.counts), A=self.A, k=self.k, beta=self.beta,
                  normalize=self.normalize, bias=self.bias)
        if self.obs_rms is not None:
            sd["obs_rms"] = (self.obs_rms.mean, self.obs_rms.var, self.obs_rms.count)
        return sd

    def load_state_dict(self, sd: dict) -> None:
        self.counts = defaultdict(int, sd["counts"])
        self.A = sd["A"]
        self.k = sd["k"]
        self.beta = sd["beta"]
        self.normalize = bool(sd.get("normalize", False))
        self.bias = bool(sd.get("bias", False))
        if self.normalize and sd.get("obs_rms") is not None:
            if self.obs_rms is None:
                self.obs_rms = RunningMeanStd(shape=sd["obs_rms"][0].shape)
            self.obs_rms.mean, self.obs_rms.var, self.obs_rms.count = sd["obs_rms"]
