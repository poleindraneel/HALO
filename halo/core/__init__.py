"""Core SDR type, abstract base classes, and shared type aliases."""

from halo.core.sdr import SDR
from halo.core.base import (
    CorticalUnitBase,
    LayerBase,
    ReliabilityModuleBase,
    ConsensusEngineBase,
)
from halo.core.types import UnitID, ReliabilityScore, DopamineSignal

__all__ = [
    "SDR",
    "CorticalUnitBase",
    "LayerBase",
    "ReliabilityModuleBase",
    "ConsensusEngineBase",
    "UnitID",
    "ReliabilityScore",
    "DopamineSignal",
]
