"""Random Network Distillation (Burda et al., 2018).

A fixed, randomly-initialized TARGET network maps states to a feature vector.
A PREDICTOR network is trained to imitate the target. On states seen often, the
predictor matches well (low error); on novel states it fails (high error). That
prediction error is the novelty bonus:

    bonus = beta * || predictor(s') - target(s') ||^2   (then normalized)

Observations and the bonus are normalized by running statistics, because RND is
sensitive to input scale and the raw error magnitude drifts as the predictor learns.
"""
from __future__ import annotations

import numpy as np
import torch

from ..networks import mlp
from .base import ExplorationBonus
from utils import RunningMeanStd


class RNDBonus(ExplorationBonus):
    name = "rnd"

    def __init__(self, obs_dim: int, cfg, device, seed: int = 0):
        torch.manual_seed(seed + 0x511)
        self.device = device
        self.beta = float(cfg.beta)
        self.normalize_obs = bool(cfg.normalize_obs)
        self.update_proportion = float(cfg.update_proportion)

        self.target = mlp([obs_dim, *cfg.hidden_sizes, cfg.feature_dim]).to(device)
        self.predictor = mlp([obs_dim, *cfg.hidden_sizes, cfg.feature_dim]).to(device)
        for p in self.target.parameters():
            p.requires_grad_(False)  # target stays frozen forever
        self.opt = torch.optim.Adam(self.predictor.parameters(), lr=cfg.lr)

        self.obs_rms = RunningMeanStd(shape=(obs_dim,))
        self.rew_rms = RunningMeanStd(shape=())
        self.last_loss = 0.0

    def _norm_obs(self, obs_t: torch.Tensor) -> torch.Tensor:
        if not self.normalize_obs:
            return obs_t
        mean = torch.as_tensor(self.obs_rms.mean, dtype=torch.float32, device=self.device)
        std = torch.as_tensor(self.obs_rms.std, dtype=torch.float32, device=self.device)
        return ((obs_t - mean) / std).clamp(-5.0, 5.0)

    def observe(self, next_obs) -> None:
        arr = np.asarray(next_obs, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        self.obs_rms.update(arr)  # track obs stats from true visitation

    @torch.no_grad()
    def compute_bonus(self, obs, act, next_obs) -> torch.Tensor:
        x = self._norm_obs(next_obs.float())
        err = ((self.predictor(x) - self.target(x)) ** 2).mean(dim=-1)  # [B]
        self.rew_rms.update(err.detach().cpu().numpy())
        return (self.beta * err / float(self.rew_rms.std)).detach()

    def update(self, batch: dict) -> dict:
        x = self._norm_obs(batch["next_obs"].float())
        with torch.no_grad():
            tgt = self.target(x)
        pred = self.predictor(x)
        per_sample = ((pred - tgt) ** 2).mean(dim=-1)
        if self.update_proportion < 1.0:  # optionally train on a random subset
            mask = (torch.rand(per_sample.shape[0], device=self.device) < self.update_proportion).float()
            loss = (per_sample * mask).sum() / torch.clamp(mask.sum(), min=1.0)
        else:
            loss = per_sample.mean()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        self.last_loss = float(loss.item())
        return dict(rnd_loss=self.last_loss)

    def diagnostics(self) -> dict:
        return dict(rnd_loss=self.last_loss)

    def state_dict(self) -> dict:
        return dict(
            predictor=self.predictor.state_dict(),
            target=self.target.state_dict(),
            opt=self.opt.state_dict(),
            obs_rms=(self.obs_rms.mean, self.obs_rms.var, self.obs_rms.count),
            rew_rms=(self.rew_rms.mean, self.rew_rms.var, self.rew_rms.count),
            last_loss=self.last_loss,
        )

    def load_state_dict(self, sd: dict) -> None:
        self.predictor.load_state_dict(sd["predictor"])
        self.target.load_state_dict(sd["target"])
        self.opt.load_state_dict(sd["opt"])
        self.obs_rms.mean, self.obs_rms.var, self.obs_rms.count = sd["obs_rms"]
        self.rew_rms.mean, self.rew_rms.var, self.rew_rms.count = sd["rew_rms"]
        self.last_loss = sd.get("last_loss", 0.0)
