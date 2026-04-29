"""Abstract base class for HALO input encoders.

Encoders convert raw input values (scalars, categories) into Sparse Distributed
Representations (SDRs) before they enter the cortical pipeline.  All concrete
encoders inherit from :class:`EncoderBase`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from halo.core.sdr import SDR

__all__ = ["EncoderBase"]


class EncoderBase(ABC):
    """Abstract encoder interface.

    An encoder maps a single input *value* to a fixed-width boolean SDR with
    exactly *w* active bits.  Semantic similarity between values is preserved
    as overlap between their SDRs.
    """

    @property
    @abstractmethod
    def n(self) -> int:
        """Total output bits."""

    @property
    @abstractmethod
    def w(self) -> int:
        """Number of active (True) bits in every output SDR."""

    @abstractmethod
    def encode(self, value: Any, *, unit_id: str = "", timestamp: int = 0) -> SDR:
        """Encode *value* into an SDR.

        Parameters
        ----------
        value:
            The input to encode (type depends on concrete subclass).
        unit_id:
            Provenance tag placed in the returned :class:`~halo.core.sdr.SDR`.
        timestamp:
            Simulation step at encoding time.

        Returns
        -------
        SDR
            Boolean SDR of length :attr:`n` with exactly :attr:`w` active bits.
        """
