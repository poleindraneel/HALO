"""Shared type aliases used throughout HALO."""

from typing import Annotated

# Identifier for a cortical unit.
UnitID = str

# Trust score in the closed interval [0.0, 1.0].
ReliabilityScore = Annotated[float, "range [0.0, 1.0]"]

# Dopaminergic reinforcement signal in the closed interval [-1.0, 1.0].
# Dopamine-modulated plasticity: Schultz 1997.
DopamineSignal = Annotated[float, "range [-1.0, 1.0]"]

__all__ = ["UnitID", "ReliabilityScore", "DopamineSignal"]
