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
    """Parameters for each CorticalUnit (HTM Spatial Pooler + learning rules).

    References
    ----------
    Hawkins et al. 2011 — Hierarchical Temporal Memory white paper.
    NeoCortexAPI (Dobric): Entities/HtmConfig.cs.
    """

    # ------------------------------------------------------------------
    # Column topology
    # ------------------------------------------------------------------
    n_columns: int          # Total minicolumns, e.g. 2048
    sparsity: float         # Target fraction of active columns, e.g. 0.02

    # ------------------------------------------------------------------
    # Receptive field
    # ------------------------------------------------------------------
    potential_radius: int   # Input radius each column can connect to; -1 = global
    potential_pct: float    # Fraction of inputs in RF to sample as potential synapses

    # ------------------------------------------------------------------
    # Permanence (proximal synapses)
    # ------------------------------------------------------------------
    syn_perm_connected: float       # Threshold above which a synapse is "connected"
    syn_perm_active_inc: float      # Permanence increment when input is active
    syn_perm_inactive_dec: float    # Permanence decrement when input is inactive
    syn_perm_max: float             # Maximum permanence (HTM spec: 1.0)

    # ------------------------------------------------------------------
    # Inhibition
    # ------------------------------------------------------------------
    stimulus_threshold: float   # Min overlap score for a column to participate
    local_area_density: float   # Target fraction of active columns in inhibition area
    global_inhibition: bool     # True → global WTA; False → local neighbourhood WTA

    # ------------------------------------------------------------------
    # Homeostatic plasticity (boosting)
    # Schultz 1997; similar mechanism in NeoCortexAPI SpatialPooler.cs
    # ------------------------------------------------------------------
    max_boost: float                    # Maximum boost factor for underactive columns
    duty_cycle_period: int              # Steps over which duty cycles are averaged
    min_pct_overlap_duty_cycles: float  # Min overlap duty cycle fraction
    min_pct_active_duty_cycles: float   # Min active duty cycle fraction
    update_period: int                  # Steps between boost/inhibition radius updates

    # ------------------------------------------------------------------
    # Deprecated — used by placeholder Hebbian rule; removed in issue #14
    # ------------------------------------------------------------------
    learning_rate: float    # Legacy Hebbian delta magnitude

    # ------------------------------------------------------------------
    # Temporal Memory (distal synapses)
    # Hawkins et al. 2011 TM spec; NeoCortexAPI TemporalMemory.cs.
    # ------------------------------------------------------------------
    cells_per_column: int = 32
    activation_threshold: int = 13      # min connected synapses to prev active cells → active segment
    min_threshold: int = 10             # min potential synapses to prev active cells → matching segment
    max_new_synapse_count: int = 20     # max new synapses grown per winner cell per step
    initial_permanence: float = 0.21    # starting permanence for newly grown synapses
    permanence_increment: float = 0.1   # TM Hebbian increment
    permanence_decrement: float = 0.1   # TM Hebbian decrement
    predicted_segment_decrement: float = 0.0  # punishment decrement for wrong predictions

    # ------------------------------------------------------------------
    # Auto-derived (set in __post_init__, do NOT put in YAML)
    # ------------------------------------------------------------------
    syn_perm_trim_threshold: float = 0.0        # active_inc / 2
    syn_perm_below_stimulus_inc: float = 0.0    # connected / 10

    def __post_init__(self) -> None:
        # Topology
        if self.n_columns < 1:
            raise ValueError(f"n_columns must be ≥ 1, got {self.n_columns}")
        if not (0.0 < self.sparsity < 1.0):
            raise ValueError(f"sparsity must be in (0, 1), got {self.sparsity}")
        if self.potential_radius < -1:
            raise ValueError(
                f"potential_radius must be -1 (global) or ≥ 0, got {self.potential_radius}"
            )
        if not (0.0 < self.potential_pct <= 1.0):
            raise ValueError(
                f"potential_pct must be in (0, 1], got {self.potential_pct}"
            )

        # Permanence ordering
        if not (0.0 < self.syn_perm_connected < 1.0):
            raise ValueError(
                f"syn_perm_connected must be in (0, 1), got {self.syn_perm_connected}"
            )
        if self.syn_perm_active_inc <= 0.0:
            raise ValueError(
                f"syn_perm_active_inc must be > 0, got {self.syn_perm_active_inc}"
            )
        if self.syn_perm_inactive_dec <= 0.0:
            raise ValueError(
                f"syn_perm_inactive_dec must be > 0, got {self.syn_perm_inactive_dec}"
            )
        if self.syn_perm_inactive_dec >= self.syn_perm_active_inc:
            raise ValueError(
                "syn_perm_inactive_dec must be < syn_perm_active_inc "
                f"(got {self.syn_perm_inactive_dec} ≥ {self.syn_perm_active_inc})"
            )
        if self.syn_perm_max != 1.0:
            raise ValueError(
                f"syn_perm_max must be 1.0 per HTM spec, got {self.syn_perm_max}"
            )

        # Inhibition
        if self.stimulus_threshold < 0.0:
            raise ValueError(
                f"stimulus_threshold must be ≥ 0, got {self.stimulus_threshold}"
            )
        if not (0.0 < self.local_area_density < 1.0):
            raise ValueError(
                f"local_area_density must be in (0, 1), got {self.local_area_density}"
            )

        # Boosting
        if self.max_boost < 1.0:
            raise ValueError(f"max_boost must be ≥ 1.0, got {self.max_boost}")
        if self.duty_cycle_period < 1:
            raise ValueError(
                f"duty_cycle_period must be ≥ 1, got {self.duty_cycle_period}"
            )
        if self.update_period < 1:
            raise ValueError(f"update_period must be ≥ 1, got {self.update_period}")
        if not (0.0 < self.min_pct_overlap_duty_cycles < 1.0):
            raise ValueError(
                "min_pct_overlap_duty_cycles must be in (0, 1), "
                f"got {self.min_pct_overlap_duty_cycles}"
            )
        if not (0.0 < self.min_pct_active_duty_cycles < 1.0):
            raise ValueError(
                "min_pct_active_duty_cycles must be in (0, 1), "
                f"got {self.min_pct_active_duty_cycles}"
            )

        # Legacy
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")

        # Temporal Memory
        if self.cells_per_column < 1:
            raise ValueError(f"cells_per_column must be ≥ 1, got {self.cells_per_column}")
        if self.min_threshold <= 0:
            raise ValueError(f"min_threshold must be > 0, got {self.min_threshold}")
        if self.activation_threshold < self.min_threshold:
            raise ValueError(
                f"activation_threshold must be ≥ min_threshold "
                f"(got {self.activation_threshold} < {self.min_threshold})"
            )
        if self.max_new_synapse_count < 1:
            raise ValueError(
                f"max_new_synapse_count must be ≥ 1, got {self.max_new_synapse_count}"
            )
        if not (0.0 <= self.initial_permanence < self.syn_perm_connected):
            raise ValueError(
                f"initial_permanence must be in [0, syn_perm_connected), "
                f"got {self.initial_permanence} (syn_perm_connected={self.syn_perm_connected})"
            )
        if self.permanence_increment < 0.0:
            raise ValueError(
                f"permanence_increment must be ≥ 0, got {self.permanence_increment}"
            )
        if self.permanence_decrement < 0.0:
            raise ValueError(
                f"permanence_decrement must be ≥ 0, got {self.permanence_decrement}"
            )
        if self.predicted_segment_decrement < 0.0:
            raise ValueError(
                f"predicted_segment_decrement must be ≥ 0, "
                f"got {self.predicted_segment_decrement}"
            )

        # Auto-derive thresholds (NeoCortexAPI HtmConfig.cs pattern)
        self.syn_perm_trim_threshold = self.syn_perm_active_inc / 2.0
        self.syn_perm_below_stimulus_inc = self.syn_perm_connected / 10.0


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
