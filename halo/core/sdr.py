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

    def intersection(self, other: "SDR") -> "SDR":
        """Bitwise AND of this SDR and *other*.

        Returns a new SDR containing only the columns active in *both*
        operands.  Complement to :meth:`union`.

        The resulting SDR inherits *self.unit_id* and *self.timestamp*.

        Parameters
        ----------
        other:
            Must have the same length *n*.
        """
        if self.n != other.n:
            raise ValueError(
                f"SDR length mismatch: {self.n} vs {other.n}"
            )
        return SDR(
            bits=np.logical_and(self.bits, other.bits),
            unit_id=self.unit_id,
            timestamp=self.timestamp,
        )

    def normalized_overlap(self, other: "SDR") -> float:
        """Scale-independent similarity in [0.0, 1.0].

        Computed as ``overlap / min(|self|, |other|)`` where |x| is the
        number of active bits in x.  Returns 0.0 if either SDR has no
        active bits.

        Parameters
        ----------
        other:
            Must have the same length *n*.
        """
        if self.n != other.n:
            raise ValueError(
                f"SDR length mismatch: {self.n} vs {other.n}"
            )
        denom = min(int(self.bits.sum()), int(other.bits.sum()))
        if denom == 0:
            return 0.0
        return float(self.overlap(other)) / float(denom)

    def hamming_distance(self, other: "SDR") -> int:
        """Number of bit positions where *self* and *other* differ (XOR sum).

        Parameters
        ----------
        other:
            Must have the same length *n*.

        Returns
        -------
        int
            0 for identical SDRs; ``n`` for fully complementary SDRs.
        """
        if self.n != other.n:
            raise ValueError(
                f"SDR length mismatch: {self.n} vs {other.n}"
            )
        return int(np.logical_xor(self.bits, other.bits).sum())

    def subsample(self, k: int, rng: np.random.Generator) -> "SDR":
        """Randomly select *k* active bits from this SDR.

        Returns a new SDR with exactly *k* active columns chosen uniformly
        at random from the current active set.

        Parameters
        ----------
        k:
            Number of active bits to retain.  Must satisfy
            ``1 <= k <= active_count``.
        rng:
            NumPy random generator for reproducibility.

        Raises
        ------
        ValueError
            If *k* is out of range.
        """
        active = self.active_indices
        if k < 1 or k > len(active):
            raise ValueError(
                f"k must be in [1, {len(active)}], got {k}"
            )
        chosen = rng.choice(active, size=k, replace=False)
        bits = np.zeros(self.n, dtype=bool)
        bits[chosen] = True
        return SDR(bits=bits, unit_id=self.unit_id, timestamp=self.timestamp)

    def noise(self, p: float, rng: np.random.Generator) -> "SDR":
        """Randomly flip *p* fraction of bits.

        Each bit position is independently flipped with probability *p*.
        Returns a new SDR; does not modify *self*.

        Parameters
        ----------
        p:
            Flip probability in [0.0, 1.0].
        rng:
            NumPy random generator for reproducibility.

        Raises
        ------
        ValueError
            If *p* is outside [0.0, 1.0].
        """
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p must be in [0.0, 1.0], got {p}")
        flip_mask = rng.random(self.n) < p
        return SDR(
            bits=np.logical_xor(self.bits, flip_mask),
            unit_id=self.unit_id,
            timestamp=self.timestamp,
        )

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

    def copy(self) -> "SDR":
        """Return a deep copy of this SDR.

        The returned SDR has an independent ``bits`` array — mutations to
        either object do not affect the other.  ``unit_id`` and ``timestamp``
        are copied verbatim.
        """
        return SDR(
            bits=self.bits.copy(),
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
