"""ReliabilityModule: dopamine-modulated trust scores.

ReliabilityModule is the *sole* owner of trust-score mutations.  No other
component may read-modify-write trust scores.

# Dopamine-modulated plasticity: Schultz 1997
"""

from __future__ import annotations

import logging

import numpy as np

from halo.config.schema import ReliabilityConfig
from halo.core.base import ReliabilityModuleBase

logger = logging.getLogger(__name__)

__all__ = ["ReliabilityModule"]


class ReliabilityModule(ReliabilityModuleBase):
    """EMA-based trust score manager with dopamine-signal updates.

    # Dopamine-modulated plasticity: Schultz 1997

    Parameters
    ----------
    unit_ids:
        All unit IDs to track.
    config:
        ReliabilityConfig specifying initial_score, alpha, min/max bounds.
    """

    def __init__(self, unit_ids: list[str], config: ReliabilityConfig) -> None:
        self._config = config
        self._scores: dict[str, float] = {
            uid: config.initial_score for uid in unit_ids
        }

    def get_score(self, unit_id: str) -> float:
        """Return the current trust score for *unit_id*.

        Raises
        ------
        KeyError
            If *unit_id* is not registered.
        """
        return self._scores[unit_id]

    def update(self, unit_id: str, signal: float) -> None:
        """Apply a dopamine signal to update the trust score via EMA.

        Score update rule::

            score ← clip(score + alpha * signal, min_score, max_score)

        # Dopamine-modulated plasticity: Schultz 1997

        Parameters
        ----------
        unit_id:
            Target unit.
        signal:
            DopamineSignal in [-1.0, 1.0].  Positive → reward; Negative → punishment.
        """
        current = self._scores[unit_id]
        updated = current + self._config.alpha * signal
        clamped = float(
            np.clip(updated, self._config.min_score, self._config.max_score)
        )
        self._scores[unit_id] = clamped
        logger.debug(
            "Reliability update %s: %.4f → %.4f (signal=%.4f)",
            unit_id,
            current,
            clamped,
            signal,
        )

    def all_scores(self) -> dict[str, float]:
        """Return a copy of all current trust scores."""
        return dict(self._scores)
