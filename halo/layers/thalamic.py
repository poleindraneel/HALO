"""Thalamic relay layer — aggregates cortical SDRs into a unified signal.

Thalamic relay: Sherman & Guillery 2006.  The thalamus does not passively
relay cortical signals; it actively shapes them.  Here we model two
aggregation modes: simple OR union and reliability-weighted combination.
"""

from __future__ import annotations

import logging

import numpy as np

from halo.config.schema import ThalamicConfig
from halo.core.base import LayerBase
from halo.core.sdr import SDR

logger = logging.getLogger(__name__)

__all__ = ["ThalamicLayer"]


class ThalamicLayer(LayerBase):
    """Thalamic relay that aggregates multiple SDRs.

    Thalamic relay: Sherman & Guillery 2006.

    Parameters
    ----------
    config:
        ThalamicConfig specifying the aggregation mode.
    """

    def __init__(self, config: ThalamicConfig) -> None:
        self._config = config

    def process(self, inputs: list[SDR]) -> list[SDR]:
        """Aggregate all input SDRs into a single relayed SDR.

        If ``aggregation == "or"``, returns a one-element list containing the
        bitwise OR of all inputs.  For ``"weighted_sum"`` without explicit
        weights, falls back to OR.

        Parameters
        ----------
        inputs:
            SDRs from cortical units (or the previous layer).

        Returns
        -------
        list[SDR]
            A one-element list containing the aggregated SDR.
        """
        if not inputs:
            logger.warning("ThalamicLayer received empty input list")
            return []
        agg = self.aggregate(inputs)
        return [agg]

    def aggregate(
        self,
        inputs: list[SDR],
        weights: dict[str, float] | None = None,
    ) -> SDR:
        """Aggregate *inputs* into a single SDR.

        Parameters
        ----------
        inputs:
            SDRs to aggregate; must all have the same length *n*.
        weights:
            Optional mapping of ``unit_id → weight`` for ``"weighted_sum"``
            mode.  Ignored in ``"or"`` mode.

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
            return SDR(bits=bits, unit_id="thalamic", timestamp=timestamp)

        # weighted_sum mode
        n = inputs[0].n
        accum = np.zeros(n, dtype=float)
        total_weight = 0.0
        for sdr in inputs:
            w = (weights or {}).get(sdr.unit_id, 1.0)
            accum += sdr.bits.astype(float) * w
            total_weight += w

        if total_weight > 0.0:
            accum /= total_weight

        bits = accum >= 0.5
        return SDR(bits=bits, unit_id="thalamic", timestamp=timestamp)

    def reset(self) -> None:
        """ThalamicLayer is stateless; nothing to reset."""
