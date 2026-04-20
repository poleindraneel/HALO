"""Sparse Distributed Representation (SDR) dataclass.

All inter-layer communication in HALO uses SDR objects — never raw numpy
arrays — so that provenance (unit_id, timestamp) is always available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["SDR"]


@dataclass
class SDR:
    """A boolean sparse distributed representation with provenance metadata.

    Attributes
    ----------
    bits:
        1-D boolean array of length *n*.  True entries are active columns.
    unit_id:
        Identifier of the cortical unit (or layer) that produced this SDR.
    timestamp:
        Simulation step counter at the time of creation.
    """

    bits: np.ndarray          # dtype bool, shape (n,)
    unit_id: str
    timestamp: int

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n(self) -> int:
        """Total number of columns (active + inactive)."""
        return int(self.bits.shape[0])

    @property
    def sparsity(self) -> float:
        """Fraction of active columns: |active| / n."""
        return float(self.bits.mean())

    @property
    def active_indices(self) -> np.ndarray:
        """Indices of active columns as a 1-D integer array."""
        return np.where(self.bits)[0]

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @staticmethod
    def empty(n: int, unit_id: str, timestamp: int) -> "SDR":
        """Return an all-zero SDR of length *n*."""
        return SDR(bits=np.zeros(n, dtype=bool), unit_id=unit_id, timestamp=timestamp)

    @staticmethod
    def from_indices(
        indices: np.ndarray,
        n: int,
        unit_id: str,
        timestamp: int,
    ) -> "SDR":
        """Construct an SDR from a list of active column indices.

        Parameters
        ----------
        indices:
            Integer array of active column positions (0-based, < n).
        n:
            Total column count.
        unit_id:
            Producing unit identifier.
        timestamp:
            Creation step.
        """
        bits = np.zeros(n, dtype=bool)
        bits[indices] = True
        return SDR(bits=bits, unit_id=unit_id, timestamp=timestamp)

    # ------------------------------------------------------------------
    # Instance operations
    # ------------------------------------------------------------------

    def overlap(self, other: "SDR") -> int:
        """Count of columns active in *both* this SDR and *other*.

        Parameters
        ----------
        other:
            Must have the same length *n*.

        Returns
        -------
        int
            Number of shared active bits.
        """
        if self.n != other.n:
            raise ValueError(
                f"SDR length mismatch: {self.n} vs {other.n}"
            )
        return int(np.logical_and(self.bits, other.bits).sum())

    def union(self, other: "SDR") -> "SDR":
        """Bitwise OR of this SDR and *other*.

        The resulting SDR uses *self.unit_id* and *self.timestamp*.
        """
        if self.n != other.n:
            raise ValueError(
                f"SDR length mismatch: {self.n} vs {other.n}"
            )
        return SDR(
            bits=np.logical_or(self.bits, other.bits),
            unit_id=self.unit_id,
            timestamp=self.timestamp,
        )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SDR(n={self.n}, active={int(self.bits.sum())}, "
            f"sparsity={self.sparsity:.4f}, unit_id={self.unit_id!r}, "
            f"timestamp={self.timestamp})"
        )
