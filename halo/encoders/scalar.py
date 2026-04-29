"""ScalarEncoder: maps continuous float values to SDRs.

Algorithm: sliding window of *w* bits across *n* output bits.  Nearby values
map to overlapping windows; distant values have little or no overlap.

References
----------
Hawkins & Ahmad 2016, "Why Neurons Have Thousands of Synapses", Fig. 1.
NeoCortexAPI: Encoders/ScalarEncoder.cs.
"""

from __future__ import annotations

import logging

import numpy as np

from halo.core.sdr import SDR
from halo.encoders.base import EncoderBase

logger = logging.getLogger(__name__)

__all__ = ["ScalarEncoder"]


class ScalarEncoder(EncoderBase):
    """Encode a continuous scalar into a fixed-width sliding-window SDR.

    Parameters
    ----------
    n:
        Total output bits.  Must be > *w*.
    w:
        Number of active bits per encoding.  Must satisfy ``1 <= w < n``.
    min_val:
        Minimum of the input range (inclusive).
    max_val:
        Maximum of the input range (inclusive for non-periodic, treated as
        equivalent to *min_val* for periodic).
    periodic:
        If *True*, the encoding wraps around so that values near *max_val*
        overlap with values near *min_val* (e.g. angle in degrees).
    """

    def __init__(
        self,
        n: int,
        w: int,
        min_val: float,
        max_val: float,
        periodic: bool = False,
    ) -> None:
        if w < 1:
            raise ValueError(f"w must be ≥ 1, got {w}")
        if n <= w:
            raise ValueError(f"n must be > w, got n={n}, w={w}")
        if min_val >= max_val:
            raise ValueError(
                f"min_val must be < max_val, got {min_val} >= {max_val}"
            )

        self._n = n
        self._w = w
        self._min_val = min_val
        self._max_val = max_val
        self._periodic = periodic

        # Non-periodic: n_buckets starting positions; periodic: n starting positions.
        self._n_buckets: int = n if periodic else n - w + 1

    # ------------------------------------------------------------------
    # EncoderBase interface
    # ------------------------------------------------------------------

    @property
    def n(self) -> int:
        return self._n

    @property
    def w(self) -> int:
        return self._w

    def encode(self, value: float, *, unit_id: str = "", timestamp: int = 0) -> SDR:
        """Encode *value* into an SDR.

        Out-of-range values are clamped for non-periodic encoders.  For
        periodic encoders the input is wrapped via modulo arithmetic.

        Parameters
        ----------
        value:
            Scalar to encode.
        unit_id:
            Provenance tag for the returned SDR.
        timestamp:
            Simulation step.
        """
        bucket = self._bucket(float(value))
        if self._periodic:
            indices = (np.arange(self._w) + bucket) % self._n
        else:
            indices = np.arange(bucket, bucket + self._w)

        logger.debug(
            "ScalarEncoder: value=%.4f → bucket=%d (periodic=%s)",
            value, bucket, self._periodic,
        )
        return SDR.from_indices(indices, self._n, unit_id, timestamp)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _bucket(self, value: float) -> int:
        """Compute the starting bucket index for *value*."""
        period = self._max_val - self._min_val

        if self._periodic:
            # Wrap into [min_val, max_val) using modulo.
            relative = (value - self._min_val) % period
            bucket = int(relative / period * self._n_buckets) % self._n_buckets
        else:
            # Clip to [0, 1] then floor-map to [0, n_buckets).
            norm = (value - self._min_val) / period
            norm = max(0.0, min(1.0, norm))
            bucket = min(int(norm * self._n_buckets), self._n_buckets - 1)

        return bucket
