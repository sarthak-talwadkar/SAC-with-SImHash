"""Pluggable exploration-bonus modules.

Each bonus implements the same interface (see base.py) so the SAC core can use
any of them -- or none -- without changing the training loop. This is what makes
the experiment a clean controlled comparison.

Use `make_bonus(name, ...)` to construct the right one from a --bonus flag.
"""
from __future__ import annotations

from .base import ExplorationBonus
from .simhash import SimHashBonus


def make_bonus(name: str, obs_dim: int, act_dim: int, cfg, device, seed: int = 0):
    """Factory: return an ExplorationBonus instance, or None for the baseline.

    `cfg` is the bonus-specific config dataclass (SimHashConfig / RNDConfig /
    ICMConfig); `name` selects which to build.
    """
    name = (name or "none").lower()
    if name == "none":
        return None
    if name == "simhash":
        return SimHashBonus(obs_dim, cfg, device, seed=seed)
    if name == "rnd":
        from .rnd import RNDBonus
        return RNDBonus(obs_dim, cfg, device, seed=seed)
    if name == "icm":
        from .icm import ICMBonus
        return ICMBonus(obs_dim, act_dim, cfg, device, seed=seed)
    raise ValueError(f"unknown bonus '{name}'; choose from none|simhash|rnd|icm")
