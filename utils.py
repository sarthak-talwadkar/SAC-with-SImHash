"""Small shared utilities: seeding, device selection, CSV logging, running stats."""
from __future__ import annotations

import csv
import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility.
    (Gym environments are seeded separately via env.reset(seed=...).)"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return the CUDA device if available (and wanted), else CPU."""
    use_cuda = prefer_cuda and torch.cuda.is_available()
    return torch.device("cuda" if use_cuda else "cpu")


class CSVLogger:
    """Append-only CSV writer. The header is inferred from the first row's keys.
    Flushes after every write so you can watch a run live (e.g. tail the file)."""

    def __init__(self, path: str, append: bool = False):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        existed = append and os.path.exists(path) and os.path.getsize(path) > 0
        self._file = open(path, "a" if append else "w", newline="")
        self._writer = None
        self._header_written = existed  # appending to a non-empty file -> header already present

    def write(self, row: dict) -> None:
        if self._writer is None:
            self._writer = csv.DictWriter(self._file, fieldnames=list(row.keys()))
            if not self._header_written:
                self._writer.writeheader()
                self._header_written = True
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()


class RunningMeanStd:
    """Online mean / variance via Welford's parallel algorithm.

    Used to normalize RND intrinsic rewards and observations: it keeps the
    bonus on a stable scale even though raw prediction errors drift as the
    predictor network learns.
    """

    def __init__(self, shape=(), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot
        self.var = m2 / tot
        self.count = tot

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var) + 1e-8
