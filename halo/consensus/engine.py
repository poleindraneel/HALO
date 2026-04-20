"""ConsensusEngine: weighted-vote SDR aggregation across cortical units."""

from __future__ import annotations

import logging

import numpy as np

from halo.config.schema import ConsensusConfig
from halo.core.base import ConsensusEngineBase
from halo.core.sdr import SDR

logger = logging.getLogger(__name__)

__all__ = ["ConsensusEngine"]


class ConsensusEngine(ConsensusEngineBase):
    """Weighted-vote consensus over multiple SDRs.

    Each bit position accumulates reliability-weighted votes from all input
    SDRs.  A bit is activated in the consensus if its weighted vote sum
    exceeds 0.5 (i.e., it represents the weighted majority).

    Parameters
    ----------
    config:
        ConsensusConfig — currently only ``method="weighted_vote"`` is
        supported.
    """

    def __init__(self, config: ConsensusConfig) -> None:
        self._config = config

    def aggregate(self, sdrs: list[SDR], weights: dict[str, float]) -> SDR:
        """Produce a consensus SDR via weighted voting.

        Parameters
        ----------
        sdrs:
            One SDR per contributing unit.  All must have the same length.
        weights:
            Mapping from ``sdr.unit_id`` to reliability weight in [0, 1].
            Units not present in the mapping receive weight 0.

        Returns
        -------
        SDR
            Consensus representation with ``unit_id="consensus"``.
        """
        if not sdrs:
            raise ValueError("Cannot aggregate empty list of SDRs")

        n = sdrs[0].n
        timestamp = max(s.timestamp for s in sdrs)
        accum = np.zeros(n, dtype=float)
        total_weight = 0.0

        for sdr in sdrs:
            w = weights.get(sdr.unit_id, 0.0)
            accum += sdr.bits.astype(float) * w
            total_weight += w

        if total_weight > 0.0:
            accum /= total_weight

        bits = accum > 0.5
        logger.debug(
            "Consensus: %d active bits from %d units (total_weight=%.4f)",
            int(bits.sum()),
            len(sdrs),
            total_weight,
        )
        return SDR(bits=bits, unit_id="consensus", timestamp=timestamp)
