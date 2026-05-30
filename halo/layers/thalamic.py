"""Thalamic relay layer — aggregates cortical SDRs into a unified signal.

Thalamic relay: Sherman & Guillery 2006.  The thalamus does not passively
relay cortical signals; it actively shapes them.  Here we model two
aggregation modes:

- ``"or"`` — bitwise union; baseline, treats all units equally.
- ``"weighted_sum"`` — reliability-weighted accumulation followed by top-k
  thresholding to maintain a target output sparsity.  Reliability weights
  are supplied via a :class:`~halo.core.base.ReliabilityModuleBase` (or a
  plain ``dict[str, float]``).  Units with higher trust scores contribute
  more to the aggregated representation.
"""

from __future__ import annotations

import logging

import numpy as np

from halo.config.schema import ThalamicConfig
from halo.core.base import LayerBase, ReliabilityModuleBase
from halo.core.sdr import SDR

logger = logging.getLogger(__name__)

__all__ = ["ThalamicLayer"]


class ThalamicLayer(LayerBase):
    """Thalamic relay that aggregates multiple SDRs.

    Thalamic relay: Sherman & Guillery 2006.

    Parameters
    ----------
    config:
        ThalamicConfig specifying aggregation mode and target output sparsity.
    """

    def __init__(self, config: ThalamicConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # LayerBase interface
    # ------------------------------------------------------------------

    def process(
        self,
        inputs: list[SDR],
        *,
        reliability: ReliabilityModuleBase | dict[str, float] | None = None,
    ) -> list[SDR]:
        """Aggregate all input SDRs into a single relayed SDR.

        Parameters
        ----------
        inputs:
            SDRs from cortical units (or the previous layer).
        reliability:
            Optional reliability source for ``"weighted_sum"`` mode.
            May be a :class:`~halo.core.base.ReliabilityModuleBase` instance
            or a plain ``dict[str, float]`` mapping unit_id → weight.
            If *None*, all units receive equal weight of 1.0.

        Returns
        -------
        list[SDR]
            A one-element list containing the aggregated SDR.
        """
        if not inputs:
            logger.warning("ThalamicLayer received empty input list")
            return []
        weights = self._resolve_weights(inputs, reliability)
        agg = self.aggregate(inputs, weights=weights)
        return [agg]

    def aggregate(
        self,
        inputs: list[SDR],
        weights: dict[str, float] | None = None,
    ) -> SDR:
        """Aggregate *inputs* into a single SDR.

        ``"or"`` mode
            Bitwise union of all inputs; *weights* are ignored.

        ``"weighted_sum"`` mode
            Each input SDR is multiplied by its unit's reliability weight and
            accumulated into a float score vector.  The top-k bits (where
            ``k = round(n * output_sparsity)``) are set active in the output.
            This preserves a stable output sparsity regardless of how many
            units contribute, and gives higher-reliability units proportionally
            more influence over which bits survive.

        Parameters
        ----------
        inputs:
            SDRs to aggregate; must all have the same length *n*.
        weights:
            Mapping of ``unit_id → weight`` for ``"weighted_sum"`` mode.
            Defaults to uniform weight 1.0 for all units if not provided.

        Returns
        -------
        SDR
            Aggregated SDR with ``unit_id="thalamic"``.
        """
        if not inputs:
            raise ValueError("Cannot aggregate empty list of SDRs")

        timestamp = max(s.timestamp for s in inputs)

        if self._config.aggregation == "or":
            bits = inputs[0].bits.copy()
            for sdr in inputs[1:]:
                bits = np.logical_or(bits, sdr.bits)
            logger.debug(
                "ThalamicLayer OR: %d inputs → %d active bits",
                len(inputs),
                int(bits.sum()),
            )
            return SDR(bits=bits, unit_id="thalamic", timestamp=timestamp)

        # ------------------------------------------------------------------
        # weighted_sum mode: reliability-weighted accumulation + top-k gate
        # ------------------------------------------------------------------
        n = inputs[0].n
        accum = np.zeros(n, dtype=float)
        total_weight = 0.0

        for sdr in inputs:
            w = 1.0 if weights is None else weights.get(sdr.unit_id, 0.0)
            accum += sdr.bits.astype(float) * w
            total_weight += w

        # Normalise so scale is independent of number of units / weight sum.
        if total_weight > 0.0:
            accum /= total_weight

        # Top-k thresholding to enforce output_sparsity.
        k = max(1, round(n * self._config.output_sparsity))
        bits = self._topk_mask(accum, k)

        logger.debug(
            "ThalamicLayer weighted_sum: %d inputs, total_weight=%.3f → %d active bits (k=%d)",
            len(inputs),
            total_weight,
            int(bits.sum()),
            k,
        )
        return SDR(bits=bits, unit_id="thalamic", timestamp=timestamp)

    def reset(self) -> None:
        """ThalamicLayer is stateless; nothing to reset."""

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_weights(
        inputs: list[SDR],
        reliability: ReliabilityModuleBase | dict[str, float] | None,
    ) -> dict[str, float]:
        """Convert a reliability source into a plain weight dict."""
        if reliability is None:
            return {sdr.unit_id: 1.0 for sdr in inputs}
        if isinstance(reliability, dict):
            return reliability
        # ReliabilityModuleBase — query each unit's score
        return {sdr.unit_id: reliability.get_score(sdr.unit_id) for sdr in inputs}

    @staticmethod
    def _topk_mask(scores: np.ndarray, k: int) -> np.ndarray:
        """Return a boolean mask with True at the top-*k* score positions.

        Ties are broken by index (lower index wins) for determinism.
        """
        if k >= len(scores):
            return np.ones(len(scores), dtype=bool)
        # argpartition gives top-k in arbitrary order; then sort for stability
        top_indices = np.argpartition(scores, -k)[-k:]
        mask = np.zeros(len(scores), dtype=bool)
        mask[top_indices] = True
        return mask
