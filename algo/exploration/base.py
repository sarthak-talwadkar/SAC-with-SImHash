"""Common interface for every exploration bonus.

The SAC training loop and update step interact with bonuses ONLY through these
three methods, so any bonus (or none) can be swapped in without touching SAC.
That interchangeability is exactly what makes the comparison controlled.
"""
from __future__ import annotations

import torch


class ExplorationBonus:
    name = "base"

    def observe(self, next_obs) -> None:
        """Called once per ENVIRONMENT step with the freshly visited state.

        Count-based methods (SimHash) use this to tally true state visitation.
        Learned methods (RND/ICM) ignore it and learn from sampled batches.
        Accepts a single state [obs_dim] or a batch [B, obs_dim].
        """
        pass

    def compute_bonus(self, obs, act, next_obs) -> torch.Tensor:
        """Return the intrinsic reward per transition, shape [B].

        Must be DETACHED (no gradient): the bonus edits the critic's reward
        target but must not backprop into the bonus's own networks.
        """
        raise NotImplementedError

    def update(self, batch: dict) -> dict:
        """Train internal predictors on a sampled batch (RND/ICM).

        No-op for SimHash (its 'model' is just an exact count table).
        Returns a dict of diagnostics (may be empty).
        """
        return {}

    def diagnostics(self) -> dict:
        """Optional extra scalars to log (e.g. number of distinct buckets)."""
        return {}

    def state_dict(self) -> dict:
        """Serialize internal state for resuming (no-op by default)."""
        return {}

    def load_state_dict(self, sd: dict) -> None:
        pass
