"""Soft Actor-Critic agent (Haarnoja et al. 2018), modern PyTorch.

Implements clipped double-Q learning, a squashed-Gaussian policy, Polyak-averaged
target critics, and automatic entropy-temperature tuning. An optional exploration
bonus edits the reward used for the critic target: r_total = r_ext + r_int.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
import torch.optim as optim

from .networks import SquashedGaussianActor, TwinCritic


class SAC:
    def __init__(self, obs_dim: int, act_dim: int, cfg, device):
        self.device = device
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # --- networks ---
        self.actor = SquashedGaussianActor(obs_dim, act_dim, cfg.hidden_sizes).to(device)
        self.critic = TwinCritic(obs_dim, act_dim, cfg.hidden_sizes).to(device)
        self.critic_target = deepcopy(self.critic).to(device)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)  # targets are updated by Polyak averaging, not gradients

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg.lr)

        # --- entropy temperature (alpha) ---
        self.autotune = cfg.autotune_alpha
        if self.autotune:
            # Target entropy heuristic: -|A| (one "nat" of randomness per action dim).
            self.target_entropy = (
                cfg.target_entropy if cfg.target_entropy is not None else -float(act_dim)
            )
            self.log_alpha = torch.tensor(
                np.log(cfg.init_alpha), dtype=torch.float32, device=device, requires_grad=True
            )
            self.alpha_opt = optim.Adam([self.log_alpha], lr=cfg.lr)
            self.alpha = float(self.log_alpha.exp().item())
        else:
            self.alpha = float(cfg.init_alpha)

    @torch.no_grad()
    def act(self, obs, deterministic: bool = False) -> np.ndarray:
        """Choose an action in [-1, 1] for a single observation."""
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        a, _ = self.actor(o, deterministic=deterministic, with_logprob=False)
        return a.squeeze(0).cpu().numpy()

    def update(self, batch: dict, bonus=None) -> dict:
        """One gradient step on critics, actor, and (optionally) temperature."""
        o, a = batch["obs"], batch["act"]
        r, no, d = batch["rew"], batch["next_obs"], batch["done"]

        # ---- exploration bonus (optional): r_total = r_ext + r_int ----
        r_int_mean = 0.0
        if bonus is not None:
            r_int = bonus.compute_bonus(o, a, no)        # detached tensor, shape [B]
            r = r + r_int
            bonus.update(batch)                          # train predictor (RND/ICM); no-op for SimHash
            r_int_mean = float(r_int.mean().item())

        # ---- critic update: regress Q toward the soft Bellman target ----
        with torch.no_grad():
            na, nlogp = self.actor(no)                   # next action ~ current policy
            q1_t, q2_t = self.critic_target(no, na)
            q_next = torch.min(q1_t, q2_t) - self.alpha * nlogp   # soft (entropy-augmented) value
            target = r + self.gamma * (1.0 - d) * q_next
        q1, q2 = self.critic(o, a)
        critic_loss = ((q1 - target) ** 2).mean() + ((q2 - target) ** 2).mean()
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        # ---- actor update: maximize Q while staying random (entropy) ----
        for p in self.critic.parameters():
            p.requires_grad_(False)                      # freeze critics during actor step (saves compute)
        pi, logp = self.actor(o)
        q1_pi, q2_pi = self.critic(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * logp - q_pi).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        for p in self.critic.parameters():
            p.requires_grad_(True)

        # ---- temperature update: push entropy toward the target ----
        if self.autotune:
            alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = float(self.log_alpha.exp().item())

        # ---- Polyak update of target critics ----
        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
                pt.data.mul_(1.0 - self.tau)
                pt.data.add_(self.tau * p.data)

        return dict(
            critic_loss=float(critic_loss.item()),
            actor_loss=float(actor_loss.item()),
            alpha=self.alpha,
            entropy=float((-logp).mean().item()),
            q_mean=float(q1.mean().item()),
            r_int_mean=r_int_mean,
        )

    # ---- checkpointing ----
    def save(self, path: str) -> None:
        torch.save(
            dict(
                actor=self.actor.state_dict(),
                critic=self.critic.state_dict(),
                critic_target=self.critic_target.state_dict(),
                alpha=self.alpha,
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
            ),
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.alpha = ckpt.get("alpha", self.alpha)

    # ---- full training state (for --resume) ----
    def state_dict(self) -> dict:
        """Everything needed to continue training: nets, optimizers, temperature."""
        sd = dict(
            actor=self.actor.state_dict(),
            critic=self.critic.state_dict(),
            critic_target=self.critic_target.state_dict(),
            actor_opt=self.actor_opt.state_dict(),
            critic_opt=self.critic_opt.state_dict(),
            alpha=self.alpha,
            autotune=self.autotune,
        )
        if self.autotune:
            sd["log_alpha"] = self.log_alpha.detach().cpu()
            sd["alpha_opt"] = self.alpha_opt.state_dict()
        return sd

    def load_state_dict(self, sd: dict) -> None:
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        self.critic_target.load_state_dict(sd["critic_target"])
        self.actor_opt.load_state_dict(sd["actor_opt"])
        self.critic_opt.load_state_dict(sd["critic_opt"])
        self.alpha = sd["alpha"]
        if self.autotune and sd.get("log_alpha") is not None:
            with torch.no_grad():
                self.log_alpha.copy_(sd["log_alpha"].to(self.device))
            self.alpha_opt.load_state_dict(sd["alpha_opt"])
            self.alpha = float(self.log_alpha.exp().item())
