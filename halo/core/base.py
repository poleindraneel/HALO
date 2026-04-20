"""Abstract base classes for all major HALO components.

Concrete implementations must inherit from these ABCs to ensure a consistent
interface across the heterarchical processing pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from halo.core.sdr import SDR

__all__ = [
    "CorticalUnitBase",
    "LayerBase",
    "ReliabilityModuleBase",
    "ConsensusEngineBase",
]


class CorticalUnitBase(ABC):
    """Abstract cortical processing unit.

    Each unit maintains its own synaptic state and produces / learns from
    Sparse Distributed Representations.
    """

    @property
    @abstractmethod
    def unit_id(self) -> str:
        """Unique identifier for this cortical unit."""
        ...

    @abstractmethod
    def encode(self, input_data: np.ndarray) -> SDR:
        """Map raw input into a Sparse Distributed Representation.

        Parameters
        ----------
        input_data:
            1-D float array of length *input_dim*.

        Returns
        -------
        SDR
            Winner-take-all sparse encoding of the input.
        """
        ...

    @abstractmethod
    def learn(self, sdr: SDR) -> None:
        """Update synaptic weights given the provided SDR.

        Hebbian-style: strengthen connections that contributed to active
        columns, weaken connections to inactive columns.

        Parameters
        ----------
        sdr:
            The SDR representing the current activation pattern.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reinitialise internal state (weights, counters, etc.)."""
        ...


class LayerBase(ABC):
    """Abstract processing layer that transforms a list of SDRs."""

    @abstractmethod
    def process(self, inputs: list[SDR]) -> list[SDR]:
        """Apply layer-specific transformation to the input SDRs.

        Parameters
        ----------
        inputs:
            One SDR per cortical unit in the previous stage.

        Returns
        -------
        list[SDR]
            Transformed (possibly gated, mixed, or aggregated) SDRs.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset any stateful buffers held by this layer."""
        ...


class ReliabilityModuleBase(ABC):
    """Abstract reliability / trust-score manager.

    ReliabilityModule is the *sole* owner of trust-score mutations.
    No other component may modify trust scores directly.
    """

    @abstractmethod
    def get_score(self, unit_id: str) -> float:
        """Return the current trust score for *unit_id* in [0, 1]."""
        ...

    @abstractmethod
    def update(self, unit_id: str, signal: float) -> None:
        """Update the trust score for *unit_id* using a dopamine signal.

        Parameters
        ----------
        unit_id:
            Target unit.
        signal:
            DopamineSignal in [-1.0, 1.0].
        """
        ...

    @abstractmethod
    def all_scores(self) -> dict[str, float]:
        """Return a snapshot of all unit trust scores."""
        ...


class ConsensusEngineBase(ABC):
    """Abstract consensus aggregator over multiple SDRs."""

    @abstractmethod
    def aggregate(self, sdrs: list[SDR], weights: dict[str, float]) -> SDR:
        """Produce a single consensus SDR from weighted unit outputs.

        Parameters
        ----------
        sdrs:
            One SDR per contributing unit.
        weights:
            Mapping from ``sdr.unit_id`` to reliability weight in [0, 1].

        Returns
        -------
        SDR
            Consensus representation (unit_id="consensus").
        """
        ...
