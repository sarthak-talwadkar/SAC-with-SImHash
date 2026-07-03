"""Neural networks for SAC: a squashed-Gaussian actor and twin Q critics.

Design choice: the actor always outputs actions in [-1, 1]. The environment
layer (envs/wrappers.py) rescales [-1, 1] to each env's true action range, so
the networks here never need to know about action bounds. This keeps the
tanh log-probability correction simple and exactly standard.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# Clamp the policy's log-std to a sane range for numerical stability.
LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity) -> nn.Sequential:
    """Build a simple multilayer perceptron from a list of layer sizes."""
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class SquashedGaussianActor(nn.Module):
    """Policy pi(a|s).

    Outputs a Gaussian over a 'pre-squash' variable u, then a = tanh(u) bounds
    the action to [-1, 1]. Sampling uses the reparameterization trick so policy
    gradients flow through the sampled action.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes=(256, 256)):
        super().__init__()
        # Trunk ends in an activation; the mean/log-std heads read its output.
        self.trunk = mlp([obs_dim, *hidden_sizes], nn.ReLU, output_activation=nn.ReLU)
        last = hidden_sizes[-1]
        self.mu_head = nn.Linear(last, act_dim)
        self.log_std_head = nn.Linear(last, act_dim)

    def forward(self, obs, deterministic: bool = False, with_logprob: bool = True):
        h = self.trunk(obs)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()

        dist = Normal(mu, std)
        # u: the pre-tanh action. At eval we use the mean (deterministic).
        u = mu if deterministic else dist.rsample()

        if with_logprob:
            # log pi(a) = log N(u) - sum log(1 - tanh(u)^2), in a numerically
            # stable form (the softplus identity avoids log of a tiny number).
            logp = dist.log_prob(u).sum(axis=-1)
            logp -= (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(axis=-1)
        else:
            logp = None

        action = torch.tanh(u)  # bounded to [-1, 1]
        return action, logp


class Critic(nn.Module):
    """A single Q-network Q(s, a) -> scalar value."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes=(256, 256)):
        super().__init__()
        self.q = mlp([obs_dim + act_dim, *hidden_sizes, 1], nn.ReLU)

    def forward(self, obs, act):
        return self.q(torch.cat([obs, act], dim=-1)).squeeze(-1)


class TwinCritic(nn.Module):
    """Two independent Q-networks. SAC uses min(Q1, Q2) to curb overestimation."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes=(256, 256)):
        super().__init__()
        self.q1 = Critic(obs_dim, act_dim, hidden_sizes)
        self.q2 = Critic(obs_dim, act_dim, hidden_sizes)

    def forward(self, obs, act):
        return self.q1(obs, act), self.q2(obs, act)
