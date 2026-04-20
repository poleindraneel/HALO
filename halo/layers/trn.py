"""Thalamic Reticular Nucleus (TRN) gating layer.

Implements entropy-based selective inhibition of cortical signals.  When the
entropy across concurrent SDRs exceeds the configured threshold, the TRN
suppresses the noisiest (least-sparse) representations.

# TRN-like selective inhibition: Crick 1984; Pinault 2004
"""

from __future__ import annotations

import logging

from halo.config.schema import TRNConfig
from halo.core.base import LayerBase
from halo.core.sdr import SDR
from halo.utils.metrics import entropy

logger = logging.getLogger(__name__)

__all__ = ["TRNGatingLayer"]


class TRNGatingLayer(LayerBase):
    """Entropy-driven selective inhibition of cortical SDRs.

    # TRN-like selective inhibition: Crick 1984; Pinault 2004

    When the Shannon entropy across the input SDR population exceeds
    ``config.entropy_threshold``, SDRs whose sparsity is below the mean
    sparsity of the population are zeroed out (replaced with an empty SDR).

    Parameters
    ----------
    config:
        TRNConfig specifying the entropy_threshold.
    """

    def __init__(self, config: TRNConfig) -> None:
        self._config = config

    def process(self, inputs: list[SDR]) -> list[SDR]:
        """Gate inputs based on population entropy.

        Parameters
        ----------
        inputs:
            SDRs from the thalamic relay (or previous layer).

        Returns
        -------
        list[SDR]
            Gated SDRs; suppressed entries are replaced with all-zero SDRs.
        """
        if not inputs:
            return []

        h = entropy(inputs)

        if h <= self._config.entropy_threshold:
            logger.debug(
                "TRN: entropy=%.4f ≤ threshold=%.4f — no gating",
                h,
                self._config.entropy_threshold,
            )
            return inputs

        logger.warning(
            "TRN gating triggered: entropy=%.4f > threshold=%.4f",
            h,
            self._config.entropy_threshold,
        )

        mean_sparsity = sum(s.sparsity for s in inputs) / len(inputs)
        gated: list[SDR] = []
        for sdr in inputs:
            if sdr.sparsity < mean_sparsity:
                # Suppress: replace with zero SDR
                gated.append(SDR.empty(sdr.n, sdr.unit_id, sdr.timestamp))
                logger.debug(
                    "TRN suppressed SDR from %s (sparsity=%.4f < mean=%.4f)",
                    sdr.unit_id,
                    sdr.sparsity,
                    mean_sparsity,
                )
            else:
                gated.append(sdr)

        return gated

    def reset(self) -> None:
        """TRNGatingLayer is stateless; nothing to reset."""
