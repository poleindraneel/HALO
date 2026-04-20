"""Pure-dataclass configuration hierarchy for HALO.

All parameters are captured here — no magic numbers elsewhere in the code.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "CorticalConfig",
    "ThalamicConfig",
    "TRNConfig",
    "ReliabilityConfig",
    "ConsensusConfig",
    "HALOConfig",
]

_VALID_AGGREGATIONS = frozenset({"or", "weighted_sum"})
_VALID_CONSENSUS_METHODS = frozenset({"weighted_vote"})


@dataclass
class CorticalConfig:
    """Parameters for each CorticalUnit."""

    n_columns: int      # e.g. 2048 — total number of minicolumns
    sparsity: float     # e.g. 0.02 — fraction of active columns per step
    learning_rate: float  # e.g. 0.1 — Hebbian weight delta magnitude

    def __post_init__(self) -> None:
        if self.n_columns < 1:
            raise ValueError(f"n_columns must be ≥ 1, got {self.n_columns}")
        if not (0.0 < self.sparsity < 1.0):
            raise ValueError(f"sparsity must be in (0, 1), got {self.sparsity}")
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")


@dataclass
class ThalamicConfig:
    """Parameters for the ThalamicLayer relay."""

    aggregation: str  # "or" | "weighted_sum"

    def __post_init__(self) -> None:
        if self.aggregation not in _VALID_AGGREGATIONS:
            raise ValueError(
                f"aggregation must be one of {_VALID_AGGREGATIONS}, "
                f"got {self.aggregation!r}"
            )


@dataclass
class TRNConfig:
    """Parameters for the TRN gating layer.

    Thalamic Reticular Nucleus — selective inhibition based on entropy.
    Crick 1984; Pinault 2004.
    """

    entropy_threshold: float  # e.g. 0.5 — gate when entropy exceeds this

    def __post_init__(self) -> None:
        if not (0.0 <= self.entropy_threshold <= 1.0):
            raise ValueError(
                f"entropy_threshold must be in [0, 1], got {self.entropy_threshold}"
            )


@dataclass
class ReliabilityConfig:
    """Parameters for the ReliabilityModule.

    Dopamine-modulated plasticity: Schultz 1997.
    """

    initial_score: float  # e.g. 0.5
    alpha: float          # EMA decay, e.g. 0.1
    min_score: float      # e.g. 0.01
    max_score: float      # e.g. 1.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.min_score <= self.initial_score <= self.max_score <= 1.0):
            raise ValueError(
                "Must satisfy 0 ≤ min_score ≤ initial_score ≤ max_score ≤ 1; "
                f"got min={self.min_score}, init={self.initial_score}, "
                f"max={self.max_score}"
            )
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")


@dataclass
class ConsensusConfig:
    """Parameters for the ConsensusEngine."""

    method: str  # "weighted_vote"

    def __post_init__(self) -> None:
        if self.method not in _VALID_CONSENSUS_METHODS:
            raise ValueError(
                f"method must be one of {_VALID_CONSENSUS_METHODS}, "
                f"got {self.method!r}"
            )


@dataclass
class HALOConfig:
    """Top-level configuration for the full HALO pipeline."""

    n_units: int
    n_input_dim: int
    cortical: CorticalConfig
    thalamic: ThalamicConfig
    trn: TRNConfig
    reliability: ReliabilityConfig
    consensus: ConsensusConfig
    max_steps: int
    seed: int

    def __post_init__(self) -> None:
        if self.n_units < 1:
            raise ValueError(f"n_units must be ≥ 1, got {self.n_units}")
        if self.n_input_dim < 1:
            raise ValueError(f"n_input_dim must be ≥ 1, got {self.n_input_dim}")
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be ≥ 1, got {self.max_steps}")
