"""Intrinsic Curiosity Module (Pathak et al., 2017).

Learn a feature space phi(s) jointly with two models:
  - INVERSE model: predict the action from (phi(s), phi(s')). This forces the
    features to encode only what the agent can control (ignores noise).
  - FORWARD model: predict phi(s') from (phi(s), a).
The agent is "curious" about transitions the forward model predicts poorly:

    bonus = beta * eta * || forward(phi(s), a) - phi(s') ||^2   (then normalized)
"""
from __future__ import annotations

import torch

from ..networks import mlp
from .base import ExplorationBonus
from utils import RunningMeanStd


class ICMBonus(ExplorationBonus):
    name = "icm"

    def __init__(self, obs_dim: int, act_dim: int, cfg, device, seed: int = 0):
        torch.manual_seed(seed + 0x1C3)
        self.device = device
        self.beta = float(cfg.beta)
        self.eta = float(cfg.eta)
        self.inverse_weight = float(cfg.inverse_weight)

        f, h = cfg.feature_dim, cfg.hidden_sizes
        self.encoder = mlp([obs_dim, *h, f]).to(device)          # phi(s)
        self.inverse = mlp([2 * f, *h, act_dim]).to(device)      # (phi,phi') -> action
        self.forward_model = mlp([f + act_dim, *h, f]).to(device)  # (phi,a) -> phi'

        params = (list(self.encoder.parameters())
                  + list(self.inverse.parameters())
                  + list(self.forward_model.parameters()))
        self.opt = torch.optim.Adam(params, lr=cfg.lr)
        self.rew_rms = RunningMeanStd(shape=())
        self.last_fwd = 0.0
        self.last_inv = 0.0

    @torch.no_grad()
    def compute_bonus(self, obs, act, next_obs) -> torch.Tensor:
        phi = self.encoder(obs.float())
        phi_next = self.encoder(next_obs.float())
        pred_next = self.forward_model(torch.cat([phi, act.float()], dim=-1))
        err = ((pred_next - phi_next) ** 2).mean(dim=-1)  # [B]
        self.rew_rms.update(err.detach().cpu().numpy())
        return (self.beta * self.eta * err / float(self.rew_rms.std)).detach()

    def update(self, batch: dict) -> dict:
        o, a, no = batch["obs"].float(), batch["act"].float(), batch["next_obs"].float()
        phi = self.encoder(o)
        phi_next = self.encoder(no)

        # inverse model: recover the action that caused the transition
        a_pred = self.inverse(torch.cat([phi, phi_next], dim=-1))
        inv_loss = ((a_pred - a) ** 2).mean()

        # forward model: predict next features (target detached -> trains forward only)
        pred_next = self.forward_model(torch.cat([phi, a], dim=-1))
        fwd_loss = ((pred_next - phi_next.detach()) ** 2).mean()

        loss = self.inverse_weight * inv_loss + (1.0 - self.inverse_weight) * fwd_loss
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        self.last_fwd = float(fwd_loss.item())
        self.last_inv = float(inv_loss.item())
        return dict(icm_fwd_loss=self.last_fwd, icm_inv_loss=self.last_inv)

    def diagnostics(self) -> dict:
        return dict(icm_fwd_loss=self.last_fwd, icm_inv_loss=self.last_inv)

    def state_dict(self) -> dict:
        return dict(
            encoder=self.encoder.state_dict(),
            inverse=self.inverse.state_dict(),
            forward=self.forward_model.state_dict(),
            opt=self.opt.state_dict(),
            rew_rms=(self.rew_rms.mean, self.rew_rms.var, self.rew_rms.count),
            last_fwd=self.last_fwd,
            last_inv=self.last_inv,
        )

    def load_state_dict(self, sd: dict) -> None:
        self.encoder.load_state_dict(sd["encoder"])
        self.inverse.load_state_dict(sd["inverse"])
        self.forward_model.load_state_dict(sd["forward"])
        self.opt.load_state_dict(sd["opt"])
        self.rew_rms.mean, self.rew_rms.var, self.rew_rms.count = sd["rew_rms"]
        self.last_fwd = sd.get("last_fwd", 0.0)
        self.last_inv = sd.get("last_inv", 0.0)
