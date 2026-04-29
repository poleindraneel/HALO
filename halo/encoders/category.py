"""CategoryEncoder: maps discrete categories to non-overlapping SDRs.

Each category receives a unique, non-overlapping block of *w* active bits.
The output width *n* must be at least ``len(categories) * w``.

References
----------
NeoCortexAPI: Encoders/CategoryEncoder.cs.
"""

from __future__ import annotations

import logging

import numpy as np

from halo.core.sdr import SDR
from halo.encoders.base import EncoderBase

logger = logging.getLogger(__name__)

__all__ = ["CategoryEncoder"]


class CategoryEncoder(EncoderBase):
    """Encode discrete category strings into non-overlapping fixed-width SDRs.

    Parameters
    ----------
    n:
        Total output bits.  Must be >= ``len(categories) * w``.
    w:
        Active bits per category.  Must be >= 1.
    categories:
        Ordered list of category labels.  Must be non-empty and have no
        duplicate entries.
    """

    def __init__(self, n: int, w: int, categories: list[str]) -> None:
        if w < 1:
            raise ValueError(f"w must be ≥ 1, got {w}")
        if not categories:
            raise ValueError("categories must be non-empty")
        if len(categories) != len(set(categories)):
            raise ValueError("categories must not contain duplicates")
        required = len(categories) * w
        if n < required:
            raise ValueError(
                f"n must be ≥ len(categories)*w = {required}, got n={n}"
            )

        self._n = n
        self._w = w
        self._categories = list(categories)
        self._cat_to_idx: dict[str, int] = {c: i for i, c in enumerate(categories)}

    # ------------------------------------------------------------------
    # EncoderBase interface
    # ------------------------------------------------------------------

    @property
    def n(self) -> int:
        return self._n

    @property
    def w(self) -> int:
        return self._w

    @property
    def categories(self) -> list[str]:
        """Ordered list of known category labels."""
        return list(self._categories)

    def encode(self, value: str, *, unit_id: str = "", timestamp: int = 0) -> SDR:
        """Encode category *value* into its unique non-overlapping SDR.

        Parameters
        ----------
        value:
            A string that must be present in :attr:`categories`.
        unit_id:
            Provenance tag for the returned SDR.
        timestamp:
            Simulation step.

        Raises
        ------
        KeyError
            If *value* is not a known category.
        """
        if value not in self._cat_to_idx:
            raise KeyError(
                f"Unknown category {value!r}. "
                f"Known: {self._categories}"
            )
        idx = self._cat_to_idx[value]
        bucket = idx * self._w
        indices = np.arange(bucket, bucket + self._w)

        logger.debug("CategoryEncoder: %r → bucket=%d", value, bucket)
        return SDR.from_indices(indices, self._n, unit_id, timestamp)
