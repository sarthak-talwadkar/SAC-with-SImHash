"""
Central configuration for the SAC + exploration-bonus experiments.

Everything that affects a run lives here, so experiments stay reproducible:
change a number once and every script (train / evaluate / plot) sees it.
`train.py` exposes the most-tuned knobs as CLI flags that override these.

We use plain dataclasses (not YAML) so the config is type-checked,
IDE-discoverable, and importable as ordinary Python.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional


# ---------------------------------------------------------------------------
# Core Soft Actor-Critic hyperparameters + training schedule
# ---------------------------------------------------------------------------
@dataclass
class SACConfig:
    # ---- network architecture ----
    hidden_sizes: tuple = (256, 256)   # MLP width for both actor and critics

    # ---- optimization ----
    lr: float = 3e-4          # Adam learning rate (actor, critics, and temperature)
    gamma: float = 0.99       # discount factor: how much the agent values the future
    tau: float = 0.005        # Polyak coeff for slow target-network tracking (0=frozen, 1=copy)
    batch_size: int = 256     # transitions sampled per gradient step

    # ---- replay buffer & exploration warmup ----
    replay_size: int = 1_000_000   # max transitions kept in memory
    start_steps: int = 5_000       # take PURELY RANDOM actions this long, to seed the buffer
    update_after: int = 1_000      # don't learn until the buffer holds at least this many
    update_every: int = 1          # env steps between learning phases
    gradient_steps: int = 1        # gradient updates performed per learning phase

    # ---- maximum-entropy temperature (alpha) ----
    # alpha trades off reward vs. policy randomness. Auto-tuning learns it to
    # hit a target entropy and removes the single most finicky SAC hyperparam.
    autotune_alpha: bool = True
    init_alpha: float = 0.2
    target_entropy: Optional[float] = None   # None -> -action_dim (the SAC default)

    # ---- training schedule ----
    total_steps: int = 1_000_000   # total environment steps for the run
    eval_every: int = 10_000       # env steps between evaluations
    eval_episodes: int = 10        # deterministic episodes averaged at each evaluation

    seed: int = 0


# ---------------------------------------------------------------------------
# Exploration-bonus configs (one per method). The bonus is added to the
# environment reward: r_total = r_extrinsic + r_intrinsic.
# ---------------------------------------------------------------------------
@dataclass
class SimHashConfig:
    """Count-based novelty via locality-sensitive hashing (Tang et al. 2017).

    The published hash sgn(A·s) is scale-invariant: codes are constant on rays
    through the origin, so k central hyperplanes carve R^d into at most
    2*sum_{i<d} C(k-1, i) regions (just 2k sectors when d=2) and magnitude
    information is lost. `normalize` and `bias` are the proposed fixes; both
    False reproduces the published formulation exactly.
    """
    k: int = 32            # code length in BITS: larger -> finer state distinctions
    beta: float = 0.05     # bonus scale:  r_int = beta / sqrt(count(hash(s')))
    normalize: bool = False  # standardize obs (running mean/std) before hashing
    bias: bool = False       # append a constant 1 -> affine hyperplanes (off-origin)


@dataclass
class RNDConfig:
    """Random Network Distillation (Burda et al. 2018): prediction-error novelty."""
    feature_dim: int = 128
    hidden_sizes: tuple = (256, 256)
    lr: float = 1e-4              # predictor-network learning rate
    beta: float = 1.0            # bonus scale, applied AFTER normalization
    normalize_obs: bool = True   # RND is scale-sensitive -> standardize observations
    update_proportion: float = 1.0  # fraction of each batch used to train the predictor


@dataclass
class ICMConfig:
    """Intrinsic Curiosity Module (Pathak et al. 2017): forward-model error."""
    feature_dim: int = 128
    hidden_sizes: tuple = (256, 256)
    lr: float = 1e-4
    beta: float = 0.2            # bonus scale on the forward-model prediction error
    eta: float = 1.0             # scaling inside the error -> bonus mapping
    inverse_weight: float = 0.8  # loss mix between inverse model (predict action) and forward model


# Map a --bonus name to its config class (None = plain SAC baseline).
BONUS_CONFIGS = {
    "none": None,
    "simhash": SimHashConfig,
    "rnd": RNDConfig,
    "icm": ICMConfig,
}


# ---------------------------------------------------------------------------
# Per-environment overrides. Small / sparse envs need fewer steps and a finer
# eval cadence than the big dense MuJoCo locomotion tasks.
# ---------------------------------------------------------------------------
ENV_OVERRIDES = {
    # --- sparse / hard-exploration ---
    "MountainCarContinuous-v0": dict(total_steps=150_000, start_steps=10_000,
                                     eval_every=5_000),
    "Pendulum-v1":              dict(total_steps=50_000, start_steps=2_000,
                                     eval_every=2_500),
    "Reacher-v5":               dict(total_steps=200_000, eval_every=5_000),
    # --- dense MuJoCo locomotion ---
    "HalfCheetah-v5":           dict(total_steps=1_000_000),
    "Hopper-v5":                dict(total_steps=1_000_000),
    "Walker2d-v5":              dict(total_steps=1_000_000),
    "Ant-v5":                   dict(total_steps=1_000_000),
}


def make_sac_config(env_id: str, **overrides) -> SACConfig:
    """Build a SACConfig: start from defaults, apply per-env overrides,
    then apply any explicit (CLI) overrides that are not None."""
    cfg = SACConfig()
    cfg = replace(cfg, **ENV_OVERRIDES.get(env_id, {}))
    clean = {k: v for k, v in overrides.items() if v is not None}
    if clean:
        cfg = replace(cfg, **clean)
    return cfg
