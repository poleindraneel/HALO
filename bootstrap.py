"""
bootstrap.py — Run once to set up the full HALO project structure.

    python bootstrap.py

Creates all directories and writes every source file of the HALO framework.
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def write(rel_path: str, content: str) -> None:
    full = os.path.join(ROOT, rel_path.replace("/", os.sep))
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  wrote {rel_path}")


FILES: dict[str, str] = {}

# ---------------------------------------------------------------------------
# halo/__init__.py
# ---------------------------------------------------------------------------
FILES["halo/__init__.py"] = '''\
"""HALO — Heterarchical Associative Learning Orchestration."""

from halo.orchestration.pipeline import HALOPipeline

__all__ = ["HALOPipeline"]
'''

# ---------------------------------------------------------------------------
# halo/core/__init__.py
# ---------------------------------------------------------------------------
FILES["halo/core/__init__.py"] = '''\
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
'''

# ---------------------------------------------------------------------------
# halo/core/types.py
# ---------------------------------------------------------------------------
FILES["halo/core/types.py"] = '''\
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
'''

# ---------------------------------------------------------------------------
# halo/core/sdr.py
# ---------------------------------------------------------------------------
FILES["halo/core/sdr.py"] = '''\
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
'''

# ---------------------------------------------------------------------------
# halo/core/base.py
# ---------------------------------------------------------------------------
FILES["halo/core/base.py"] = '''\
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
'''

# ---------------------------------------------------------------------------
# halo/config/__init__.py
# ---------------------------------------------------------------------------
FILES["halo/config/__init__.py"] = '''\
"""Configuration schema and YAML loader for HALO."""

from halo.config.schema import (
    CorticalConfig,
    ThalamicConfig,
    TRNConfig,
    ReliabilityConfig,
    ConsensusConfig,
    HALOConfig,
)
from halo.config.loader import load_config

__all__ = [
    "CorticalConfig",
    "ThalamicConfig",
    "TRNConfig",
    "ReliabilityConfig",
    "ConsensusConfig",
    "HALOConfig",
    "load_config",
]
'''

# ---------------------------------------------------------------------------
# halo/config/schema.py
# ---------------------------------------------------------------------------
FILES["halo/config/schema.py"] = '''\
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
'''

# ---------------------------------------------------------------------------
# halo/config/loader.py
# ---------------------------------------------------------------------------
FILES["halo/config/loader.py"] = '''\
"""YAML → HALOConfig loader."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from halo.config.schema import (
    CorticalConfig,
    ConsensusConfig,
    HALOConfig,
    ReliabilityConfig,
    ThalamicConfig,
    TRNConfig,
)

logger = logging.getLogger(__name__)

__all__ = ["load_config"]


def load_config(path: str | Path) -> HALOConfig:
    """Parse *path* (YAML) and return a fully validated :class:`HALOConfig`.

    Parameters
    ----------
    path:
        Absolute or relative path to a YAML configuration file.

    Returns
    -------
    HALOConfig
        Nested, validated configuration object.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    KeyError
        If a required top-level key is missing.
    ValueError
        If any config value fails :class:`~halo.config.schema` validation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    logger.debug("Loading HALO config from %s", path)
    with path.open("r", encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh)

    cortical = CorticalConfig(**raw["cortical"])
    thalamic = ThalamicConfig(**raw["thalamic"])
    trn = TRNConfig(**raw["trn"])
    reliability = ReliabilityConfig(**raw["reliability"])
    consensus = ConsensusConfig(**raw["consensus"])

    cfg = HALOConfig(
        n_units=int(raw["n_units"]),
        n_input_dim=int(raw["n_input_dim"]),
        cortical=cortical,
        thalamic=thalamic,
        trn=trn,
        reliability=reliability,
        consensus=consensus,
        max_steps=int(raw["max_steps"]),
        seed=int(raw["seed"]),
    )
    logger.info("Config loaded: %d units, %d steps", cfg.n_units, cfg.max_steps)
    return cfg
'''

# ---------------------------------------------------------------------------
# halo/utils/__init__.py
# ---------------------------------------------------------------------------
FILES["halo/utils/__init__.py"] = '''\
"""Utility helpers: logging, metrics, serialization."""

from halo.utils.logging import get_logger
from halo.utils.metrics import sparsity, overlap_score, entropy
from halo.utils.serialization import save_state, load_state

__all__ = [
    "get_logger",
    "sparsity",
    "overlap_score",
    "entropy",
    "save_state",
    "load_state",
]
'''

# ---------------------------------------------------------------------------
# halo/utils/logging.py
# ---------------------------------------------------------------------------
FILES["halo/utils/logging.py"] = '''\
"""Centralised logger factory for HALO modules."""

import logging
import sys

__all__ = ["get_logger"]

_HANDLER_ATTACHED: set[str] = set()


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with a standard StreamHandler if none is configured.

    Calling :func:`logging.getLogger` directly is also fine inside modules;
    this helper ensures at least one handler is present so messages are not
    silently discarded.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    log = logging.getLogger(name)
    # Only attach a handler once per root-name to avoid duplicate output.
    root_name = name.split(".")[0]
    if root_name not in _HANDLER_ATTACHED and not logging.root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
        )
        log.addHandler(handler)
        _HANDLER_ATTACHED.add(root_name)
    return log
'''

# ---------------------------------------------------------------------------
# halo/utils/metrics.py
# ---------------------------------------------------------------------------
FILES["halo/utils/metrics.py"] = '''\
"""Scalar metrics computed over SDRs."""

from __future__ import annotations

import logging

import numpy as np

from halo.core.sdr import SDR

logger = logging.getLogger(__name__)

__all__ = ["sparsity", "overlap_score", "entropy"]


def sparsity(sdr: SDR) -> float:
    """Return the fraction of active columns in *sdr*."""
    return sdr.sparsity


def overlap_score(a: SDR, b: SDR) -> float:
    """Normalised overlap: shared_bits / min(|a|, |b|).

    Returns 0.0 if either SDR has no active bits.
    """
    shared = a.overlap(b)
    denom = min(int(a.bits.sum()), int(b.bits.sum()))
    if denom == 0:
        return 0.0
    return float(shared) / float(denom)


def entropy(sdrs: list[SDR]) -> float:
    """Shannon entropy over column activation frequencies across *sdrs*.

    Computes the per-column activation probability across all SDRs, then
    returns the mean binary entropy H(p) = -p log2(p) - (1-p) log2(1-p).

    Returns 0.0 for an empty list.
    """
    if not sdrs:
        return 0.0

    stacked = np.stack([s.bits.astype(float) for s in sdrs], axis=0)  # (T, n)
    p = stacked.mean(axis=0)  # per-column activation probability

    # Avoid log(0): clip to (eps, 1-eps)
    eps = 1e-10
    p_clipped = np.clip(p, eps, 1.0 - eps)
    h = -p_clipped * np.log2(p_clipped) - (1.0 - p_clipped) * np.log2(1.0 - p_clipped)
    return float(h.mean())
'''

# ---------------------------------------------------------------------------
# halo/utils/serialization.py
# ---------------------------------------------------------------------------
FILES["halo/utils/serialization.py"] = '''\
"""JSON serialization helpers that handle numpy arrays."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["save_state", "load_state"]


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


def save_state(obj: Any, path: str | Path) -> None:
    """Serialize *obj* state to JSON, converting numpy arrays to lists.

    Parameters
    ----------
    obj:
        Any JSON-serialisable object (dicts, lists, numpy arrays).
    path:
        Destination file path; parent directory must exist.
    """
    path = Path(path)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, cls=_NumpyEncoder, indent=2)
    logger.debug("State saved to %s", path)


def load_state(path: str | Path) -> Any:
    """Deserialise state from JSON produced by :func:`save_state`.

    Parameters
    ----------
    path:
        Path to the JSON file.

    Returns
    -------
    Any
        The deserialised Python object.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    logger.debug("State loaded from %s", path)
    return data
'''

# ---------------------------------------------------------------------------
# halo/models/__init__.py
# ---------------------------------------------------------------------------
FILES["halo/models/__init__.py"] = '''\
"""Cortical unit model."""

from halo.models.cortical_unit import CorticalUnit

__all__ = ["CorticalUnit"]
'''

# ---------------------------------------------------------------------------
# halo/models/cortical_unit.py
# ---------------------------------------------------------------------------
FILES["halo/models/cortical_unit.py"] = '''\
"""CorticalUnit: SDR encoder with Hebbian learning.

Implements a simplified model of a cortical minicolumn ensemble.  Input is
projected through a random synaptic weight matrix; the top-k activations
(winner-take-all) form the SDR.  Weights are updated by a local Hebbian rule —
no gradient computation, no backpropagation.
"""

from __future__ import annotations

import logging

import numpy as np

from halo.config.schema import CorticalConfig
from halo.core.base import CorticalUnitBase
from halo.core.sdr import SDR

logger = logging.getLogger(__name__)

__all__ = ["CorticalUnit"]


class CorticalUnit(CorticalUnitBase):
    """Cortical unit with winner-take-all encoding and Hebbian weight updates.

    Parameters
    ----------
    unit_id:
        Unique identifier string, e.g. ``"unit_0"``.
    config:
        CorticalConfig supplying n_columns, sparsity, learning_rate.
    rng:
        NumPy random generator for reproducible initialisation.
    input_dim:
        Dimensionality of the raw input vector.
    """

    def __init__(
        self,
        unit_id: str,
        config: CorticalConfig,
        rng: np.random.Generator,
        input_dim: int,
    ) -> None:
        self._unit_id = unit_id
        self._config = config
        self._rng = rng
        self._input_dim = input_dim
        self._step: int = 0
        self._init_weights()

    # ------------------------------------------------------------------
    # CorticalUnitBase interface
    # ------------------------------------------------------------------

    @property
    def unit_id(self) -> str:
        return self._unit_id

    def encode(self, input_data: np.ndarray) -> SDR:
        """Project *input_data* through synaptic weights and apply WTA.

        Parameters
        ----------
        input_data:
            Float array of shape ``(input_dim,)``.

        Returns
        -------
        SDR
            Sparse encoding with ``sparsity * n_columns`` active bits.
        """
        activations = self._synaptic_weights @ input_data  # (n_columns,)
        k = max(1, int(self._config.sparsity * self._config.n_columns))
        bits = self._winner_take_all(activations, k)
        self._step += 1
        return SDR(bits=bits, unit_id=self._unit_id, timestamp=self._step)

    def learn(self, sdr: SDR) -> None:
        """Hebbian weight update: strengthen active, weaken inactive columns.

        Active columns (bits==True) receive a positive delta; inactive
        columns receive a negative delta scaled by ``learning_rate``.

        This is a simplified Oja-style rule without explicit normalisation —
        sufficient for demonstration purposes.
        """
        delta = np.where(
            sdr.bits[:, np.newaxis],           # (n_columns, 1) broadcast
            self._config.learning_rate,
            -self._config.learning_rate * 0.1,  # mild decay for inactive
        )
        self._synaptic_weights += delta
        logger.debug(
            "%s learned from SDR with %d active columns",
            self._unit_id,
            int(sdr.bits.sum()),
        )

    def reset(self) -> None:
        """Reinitialise synaptic weights and step counter."""
        self._step = 0
        self._init_weights()
        logger.debug("%s reset", self._unit_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        self._synaptic_weights: np.ndarray = self._rng.uniform(
            low=0.0,
            high=1.0,
            size=(self._config.n_columns, self._input_dim),
        )

    def _winner_take_all(self, activations: np.ndarray, k: int) -> np.ndarray:
        """Return a boolean mask with exactly *k* True entries at the top-k positions.

        Parameters
        ----------
        activations:
            Float array of shape ``(n_columns,)``.
        k:
            Number of winners.

        Returns
        -------
        np.ndarray
            Boolean array of shape ``(n_columns,)``.
        """
        indices = np.argpartition(activations, -k)[-k:]
        mask = np.zeros(len(activations), dtype=bool)
        mask[indices] = True
        return mask
'''

# ---------------------------------------------------------------------------
# halo/layers/__init__.py
# ---------------------------------------------------------------------------
FILES["halo/layers/__init__.py"] = '''\
"""Processing layers: heterarchical lateral mixing, thalamic relay, TRN gating."""

from halo.layers.heterarchical import HeterarchicalLayer
from halo.layers.thalamic import ThalamicLayer
from halo.layers.trn import TRNGatingLayer

__all__ = ["HeterarchicalLayer", "ThalamicLayer", "TRNGatingLayer"]
'''

# ---------------------------------------------------------------------------
# halo/layers/heterarchical.py
# ---------------------------------------------------------------------------
FILES["halo/layers/heterarchical.py"] = '''\
"""Heterarchical (non-hierarchical peer-to-peer) lateral mixing layer.

Units at the same processing level can influence each other via lateral
connections — forming a heterarchy rather than a strict hierarchy.  This
models horizontal cortico-cortical connections.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from halo.core.base import LayerBase
from halo.core.sdr import SDR

logger = logging.getLogger(__name__)

__all__ = ["HeterarchicalLayer"]


class HeterarchicalLayer(LayerBase):
    """Lateral SDR mixing across peer cortical units.

    Usage
    -----
    >>> layer = HeterarchicalLayer()
    >>> layer.register_unit("unit_0")
    >>> layer.register_unit("unit_1")
    >>> layer.add_connection("unit_0", "unit_1")
    >>> mixed = layer.process(sdrs)
    """

    def __init__(self) -> None:
        # adjacency: from_id → list of to_ids
        self._adjacency: dict[str, list[str]] = defaultdict(list)
        self._units: set[str] = set()

    def register_unit(self, unit_id: str) -> None:
        """Register *unit_id* as a node in the heterarchical graph."""
        self._units.add(unit_id)
        logger.debug("Registered unit %s", unit_id)

    def add_connection(self, from_id: str, to_id: str) -> None:
        """Add a directed lateral connection from *from_id* to *to_id*.

        Parameters
        ----------
        from_id:
            Source unit — its SDR will influence *to_id*.
        to_id:
            Target unit — receives lateral signal from *from_id*.
        """
        self._adjacency[from_id].append(to_id)
        logger.debug("Lateral connection %s → %s", from_id, to_id)

    def process(self, inputs: list[SDR]) -> list[SDR]:
        """Mix each SDR with lateral signals from its connected peers.

        For each SDR, the union of all incoming SDRs (from units that have a
        directed connection *to* this unit) is computed and OR-ed with the
        original SDR.

        Parameters
        ----------
        inputs:
            One SDR per registered unit.

        Returns
        -------
        list[SDR]
            Laterally modified SDRs in the same order as *inputs*.
        """
        # Build a lookup: unit_id → SDR
        sdr_map: dict[str, SDR] = {sdr.unit_id: sdr for sdr in inputs}

        # Build reverse adjacency: to_id → [from_ids]
        reverse: dict[str, list[str]] = defaultdict(list)
        for from_id, to_ids in self._adjacency.items():
            for to_id in to_ids:
                reverse[to_id].append(from_id)

        result: list[SDR] = []
        for sdr in inputs:
            incoming = [
                sdr_map[fid]
                for fid in reverse.get(sdr.unit_id, [])
                if fid in sdr_map
            ]
            mixed = sdr
            for lateral in incoming:
                mixed = mixed.union(lateral)
            result.append(mixed)

        return result

    def reset(self) -> None:
        """Clear all connections and registered units."""
        self._adjacency.clear()
        self._units.clear()
'''

# ---------------------------------------------------------------------------
# halo/layers/thalamic.py
# ---------------------------------------------------------------------------
FILES["halo/layers/thalamic.py"] = '''\
"""Thalamic relay layer — aggregates cortical SDRs into a unified signal.

Thalamic relay: Sherman & Guillery 2006.  The thalamus does not passively
relay cortical signals; it actively shapes them.  Here we model two
aggregation modes: simple OR union and reliability-weighted combination.
"""

from __future__ import annotations

import logging

import numpy as np

from halo.config.schema import ThalamicConfig
from halo.core.base import LayerBase
from halo.core.sdr import SDR

logger = logging.getLogger(__name__)

__all__ = ["ThalamicLayer"]


class ThalamicLayer(LayerBase):
    """Thalamic relay that aggregates multiple SDRs.

    Thalamic relay: Sherman & Guillery 2006.

    Parameters
    ----------
    config:
        ThalamicConfig specifying the aggregation mode.
    """

    def __init__(self, config: ThalamicConfig) -> None:
        self._config = config

    def process(self, inputs: list[SDR]) -> list[SDR]:
        """Aggregate all input SDRs into a single relayed SDR.

        If ``aggregation == "or"``, returns a one-element list containing the
        bitwise OR of all inputs.  For ``"weighted_sum"`` without explicit
        weights, falls back to OR.

        Parameters
        ----------
        inputs:
            SDRs from cortical units (or the previous layer).

        Returns
        -------
        list[SDR]
            A one-element list containing the aggregated SDR.
        """
        if not inputs:
            logger.warning("ThalamicLayer received empty input list")
            return []
        agg = self.aggregate(inputs)
        return [agg]

    def aggregate(
        self,
        inputs: list[SDR],
        weights: dict[str, float] | None = None,
    ) -> SDR:
        """Aggregate *inputs* into a single SDR.

        Parameters
        ----------
        inputs:
            SDRs to aggregate; must all have the same length *n*.
        weights:
            Optional mapping of ``unit_id → weight`` for ``"weighted_sum"``
            mode.  Ignored in ``"or"`` mode.

        Returns
        -------
        SDR
            Aggregated SDR with ``unit_id="thalamic"``.
        """
        if not inputs:
            raise ValueError("Cannot aggregate empty list of SDRs")

        timestamp = max(s.timestamp for s in inputs)

        if self._config.aggregation == "or":
            bits = inputs[0].bits.copy()
            for sdr in inputs[1:]:
                bits = np.logical_or(bits, sdr.bits)
            return SDR(bits=bits, unit_id="thalamic", timestamp=timestamp)

        # weighted_sum mode
        n = inputs[0].n
        accum = np.zeros(n, dtype=float)
        total_weight = 0.0
        for sdr in inputs:
            w = (weights or {}).get(sdr.unit_id, 1.0)
            accum += sdr.bits.astype(float) * w
            total_weight += w

        if total_weight > 0.0:
            accum /= total_weight

        bits = accum >= 0.5
        return SDR(bits=bits, unit_id="thalamic", timestamp=timestamp)

    def reset(self) -> None:
        """ThalamicLayer is stateless; nothing to reset."""
'''

# ---------------------------------------------------------------------------
# halo/layers/trn.py
# ---------------------------------------------------------------------------
FILES["halo/layers/trn.py"] = '''\
"""Thalamic Reticular Nucleus (TRN) gating layer.

Implements entropy-based selective inhibition of cortical signals.  When the
entropy across concurrent SDRs exceeds the configured threshold, the TRN
suppresses the noisiest (least-sparse) representations.

# TRN-like selective inhibition: Crick 1984; Pinault 2004
"""

from __future__ import annotations

import logging

from halo.config.schema import TRNConfig
from halo.core.base import LayerBase
from halo.core.sdr import SDR
from halo.utils.metrics import entropy

logger = logging.getLogger(__name__)

__all__ = ["TRNGatingLayer"]


class TRNGatingLayer(LayerBase):
    """Entropy-driven selective inhibition of cortical SDRs.

    # TRN-like selective inhibition: Crick 1984; Pinault 2004

    When the Shannon entropy across the input SDR population exceeds
    ``config.entropy_threshold``, SDRs whose sparsity is below the mean
    sparsity of the population are zeroed out (replaced with an empty SDR).

    Parameters
    ----------
    config:
        TRNConfig specifying the entropy_threshold.
    """

    def __init__(self, config: TRNConfig) -> None:
        self._config = config

    def process(self, inputs: list[SDR]) -> list[SDR]:
        """Gate inputs based on population entropy.

        Parameters
        ----------
        inputs:
            SDRs from the thalamic relay (or previous layer).

        Returns
        -------
        list[SDR]
            Gated SDRs; suppressed entries are replaced with all-zero SDRs.
        """
        if not inputs:
            return []

        h = entropy(inputs)

        if h <= self._config.entropy_threshold:
            logger.debug(
                "TRN: entropy=%.4f ≤ threshold=%.4f — no gating",
                h,
                self._config.entropy_threshold,
            )
            return inputs

        logger.warning(
            "TRN gating triggered: entropy=%.4f > threshold=%.4f",
            h,
            self._config.entropy_threshold,
        )

        mean_sparsity = sum(s.sparsity for s in inputs) / len(inputs)
        gated: list[SDR] = []
        for sdr in inputs:
            if sdr.sparsity < mean_sparsity:
                # Suppress: replace with zero SDR
                gated.append(SDR.empty(sdr.n, sdr.unit_id, sdr.timestamp))
                logger.debug(
                    "TRN suppressed SDR from %s (sparsity=%.4f < mean=%.4f)",
                    sdr.unit_id,
                    sdr.sparsity,
                    mean_sparsity,
                )
            else:
                gated.append(sdr)

        return gated

    def reset(self) -> None:
        """TRNGatingLayer is stateless; nothing to reset."""
'''

# ---------------------------------------------------------------------------
# halo/reliability/__init__.py
# ---------------------------------------------------------------------------
FILES["halo/reliability/__init__.py"] = '''\
"""Reliability (trust-score) module — sole owner of trust-score mutations."""

from halo.reliability.module import ReliabilityModule

__all__ = ["ReliabilityModule"]
'''

# ---------------------------------------------------------------------------
# halo/reliability/module.py
# ---------------------------------------------------------------------------
FILES["halo/reliability/module.py"] = '''\
"""ReliabilityModule: dopamine-modulated trust scores.

ReliabilityModule is the *sole* owner of trust-score mutations.  No other
component may read-modify-write trust scores.

# Dopamine-modulated plasticity: Schultz 1997
"""

from __future__ import annotations

import logging

import numpy as np

from halo.config.schema import ReliabilityConfig
from halo.core.base import ReliabilityModuleBase

logger = logging.getLogger(__name__)

__all__ = ["ReliabilityModule"]


class ReliabilityModule(ReliabilityModuleBase):
    """EMA-based trust score manager with dopamine-signal updates.

    # Dopamine-modulated plasticity: Schultz 1997

    Parameters
    ----------
    unit_ids:
        All unit IDs to track.
    config:
        ReliabilityConfig specifying initial_score, alpha, min/max bounds.
    """

    def __init__(self, unit_ids: list[str], config: ReliabilityConfig) -> None:
        self._config = config
        self._scores: dict[str, float] = {
            uid: config.initial_score for uid in unit_ids
        }

    def get_score(self, unit_id: str) -> float:
        """Return the current trust score for *unit_id*.

        Raises
        ------
        KeyError
            If *unit_id* is not registered.
        """
        return self._scores[unit_id]

    def update(self, unit_id: str, signal: float) -> None:
        """Apply a dopamine signal to update the trust score via EMA.

        Score update rule::

            score ← clip(score + alpha * signal, min_score, max_score)

        # Dopamine-modulated plasticity: Schultz 1997

        Parameters
        ----------
        unit_id:
            Target unit.
        signal:
            DopamineSignal in [-1.0, 1.0].  Positive → reward; Negative → punishment.
        """
        current = self._scores[unit_id]
        updated = current + self._config.alpha * signal
        clamped = float(
            np.clip(updated, self._config.min_score, self._config.max_score)
        )
        self._scores[unit_id] = clamped
        logger.debug(
            "Reliability update %s: %.4f → %.4f (signal=%.4f)",
            unit_id,
            current,
            clamped,
            signal,
        )

    def all_scores(self) -> dict[str, float]:
        """Return a copy of all current trust scores."""
        return dict(self._scores)
'''

# ---------------------------------------------------------------------------
# halo/consensus/__init__.py
# ---------------------------------------------------------------------------
FILES["halo/consensus/__init__.py"] = '''\
"""Consensus engine — weighted-vote SDR aggregation."""

from halo.consensus.engine import ConsensusEngine

__all__ = ["ConsensusEngine"]
'''

# ---------------------------------------------------------------------------
# halo/consensus/engine.py
# ---------------------------------------------------------------------------
FILES["halo/consensus/engine.py"] = '''\
"""ConsensusEngine: weighted-vote SDR aggregation across cortical units."""

from __future__ import annotations

import logging

import numpy as np

from halo.config.schema import ConsensusConfig
from halo.core.base import ConsensusEngineBase
from halo.core.sdr import SDR

logger = logging.getLogger(__name__)

__all__ = ["ConsensusEngine"]


class ConsensusEngine(ConsensusEngineBase):
    """Weighted-vote consensus over multiple SDRs.

    Each bit position accumulates reliability-weighted votes from all input
    SDRs.  A bit is activated in the consensus if its weighted vote sum
    exceeds 0.5 (i.e., it represents the weighted majority).

    Parameters
    ----------
    config:
        ConsensusConfig — currently only ``method="weighted_vote"`` is
        supported.
    """

    def __init__(self, config: ConsensusConfig) -> None:
        self._config = config

    def aggregate(self, sdrs: list[SDR], weights: dict[str, float]) -> SDR:
        """Produce a consensus SDR via weighted voting.

        Parameters
        ----------
        sdrs:
            One SDR per contributing unit.  All must have the same length.
        weights:
            Mapping from ``sdr.unit_id`` to reliability weight in [0, 1].
            Units not present in the mapping receive weight 0.

        Returns
        -------
        SDR
            Consensus representation with ``unit_id="consensus"``.
        """
        if not sdrs:
            raise ValueError("Cannot aggregate empty list of SDRs")

        n = sdrs[0].n
        timestamp = max(s.timestamp for s in sdrs)
        accum = np.zeros(n, dtype=float)
        total_weight = 0.0

        for sdr in sdrs:
            w = weights.get(sdr.unit_id, 0.0)
            accum += sdr.bits.astype(float) * w
            total_weight += w

        if total_weight > 0.0:
            accum /= total_weight

        bits = accum > 0.5
        logger.debug(
            "Consensus: %d active bits from %d units (total_weight=%.4f)",
            int(bits.sum()),
            len(sdrs),
            total_weight,
        )
        return SDR(bits=bits, unit_id="consensus", timestamp=timestamp)
'''

# ---------------------------------------------------------------------------
# halo/orchestration/__init__.py
# ---------------------------------------------------------------------------
FILES["halo/orchestration/__init__.py"] = '''\
"""End-to-end HALO orchestration pipeline."""

from halo.orchestration.pipeline import HALOPipeline

__all__ = ["HALOPipeline"]
'''

# ---------------------------------------------------------------------------
# halo/orchestration/pipeline.py
# ---------------------------------------------------------------------------
FILES["halo/orchestration/pipeline.py"] = '''\
"""HALOPipeline: orchestrates the full biologically inspired processing loop.

Step sequence per timestep
--------------------------
1. Each CorticalUnit encodes the raw input → list[SDR]
2. HeterarchicalLayer performs lateral mixing → list[SDR]
3. Each CorticalUnit learns from its (possibly mixed) SDR
4. ThalamicLayer aggregates mixed SDRs → broadcast signal (single SDR)
5. TRNGatingLayer filters per-unit mixed SDRs by population entropy → list[SDR]
6. ConsensusEngine produces final SDR from gated per-unit SDRs
7. Dopamine signal = overlap score between consecutive outputs
8. ReliabilityModule updates all unit scores
9. Increment step counter; return final SDR
"""

from __future__ import annotations

import logging

import numpy as np

from halo.config.schema import HALOConfig
from halo.consensus.engine import ConsensusEngine
from halo.core.sdr import SDR
from halo.layers.heterarchical import HeterarchicalLayer
from halo.layers.thalamic import ThalamicLayer
from halo.layers.trn import TRNGatingLayer
from halo.models.cortical_unit import CorticalUnit
from halo.reliability.module import ReliabilityModule
from halo.utils.metrics import overlap_score

logger = logging.getLogger(__name__)

__all__ = ["HALOPipeline"]


class HALOPipeline:
    """Full HALO processing pipeline.

    Parameters
    ----------
    config:
        Top-level :class:`~halo.config.schema.HALOConfig`.
    """

    def __init__(self, config: HALOConfig) -> None:
        self._config = config
        self._step: int = 0
        self._reliability_history: list[dict[str, float]] = []

        rng = np.random.default_rng(config.seed)

        # --- Cortical units ---
        self._unit_ids: list[str] = [f"unit_{i}" for i in range(config.n_units)]
        self._units: list[CorticalUnit] = [
            CorticalUnit(
                unit_id=uid,
                config=config.cortical,
                rng=np.random.default_rng(rng.integers(2**31)),
                input_dim=config.n_input_dim,
            )
            for uid in self._unit_ids
        ]

        # --- Heterarchical layer (all-to-all) ---
        self._heterarchical = HeterarchicalLayer()
        for uid in self._unit_ids:
            self._heterarchical.register_unit(uid)
        for i, from_id in enumerate(self._unit_ids):
            for j, to_id in enumerate(self._unit_ids):
                if i != j:
                    self._heterarchical.add_connection(from_id, to_id)

        # --- Thalamic relay ---
        self._thalamic = ThalamicLayer(config.thalamic)

        # --- TRN gating ---
        self._trn = TRNGatingLayer(config.trn)

        # --- Reliability ---
        self._reliability = ReliabilityModule(self._unit_ids, config.reliability)

        # --- Consensus ---
        self._consensus = ConsensusEngine(config.consensus)

        # Previous output for dopamine signal computation
        self._prev_output: SDR | None = None

        logger.info(
            "HALOPipeline initialised: %d units, input_dim=%d",
            config.n_units,
            config.n_input_dim,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, input_data: np.ndarray) -> SDR:
        """Execute one processing step.

        Parameters
        ----------
        input_data:
            Float array of shape ``(n_input_dim,)``.

        Returns
        -------
        SDR
            Consensus SDR for this timestep.
        """
        # 1. Encode
        raw_sdrs: list[SDR] = [unit.encode(input_data) for unit in self._units]

        # 2. Lateral mixing
        mixed_sdrs = self._heterarchical.process(raw_sdrs)

        # 3. Learn from mixed SDR
        for unit, sdr in zip(self._units, mixed_sdrs):
            unit.learn(sdr)

        # 4. Thalamic relay — aggregate mixed SDRs into a single broadcast signal.
        #    (Thalamic relay: Sherman & Guillery 2006)
        _thalamic_broadcast = self._thalamic.process(mixed_sdrs)  # [SDR(unit_id="thalamic")]

        # 5. TRN gates the per-unit mixed SDRs based on population entropy.
        #    (TRN-like selective inhibition: Crick 1984; Pinault 2004)
        #    Gating on per-unit SDRs preserves unit_id for downstream weighting.
        gated = self._trn.process(mixed_sdrs)

        # 6. Consensus over gated per-unit SDRs weighted by reliability scores.
        scores = self._reliability.all_scores()
        if gated:
            final_sdr = self._consensus.aggregate(gated, scores)
        else:
            # Fallback if all SDRs were suppressed by TRN.
            final_sdr = _thalamic_broadcast[0] if _thalamic_broadcast else SDR.empty(
                self._config.cortical.n_columns, "consensus", self._step
            )

        # 7. Dopamine signal: overlap with previous output
        if self._prev_output is not None and self._prev_output.n == final_sdr.n:
            dopamine = overlap_score(self._prev_output, final_sdr) * 2.0 - 1.0
        else:
            dopamine = 0.0

        # 8. Update reliability scores (broadcast same signal to all units)
        for uid in self._unit_ids:
            self._reliability.update(uid, dopamine)
        self._reliability_history.append(self._reliability.all_scores())

        self._prev_output = final_sdr
        self._step += 1

        logger.debug(
            "Step %d: final SDR active=%d, dopamine=%.4f",
            self._step,
            int(final_sdr.bits.sum()),
            dopamine,
        )
        return final_sdr

    def run(
        self,
        input_stream: list[np.ndarray] | None = None,
    ) -> list[SDR]:
        """Run the pipeline for ``config.max_steps`` steps.

        Parameters
        ----------
        input_stream:
            Optional list of input arrays.  If *None* or shorter than
            ``max_steps``, random inputs are generated for missing steps.

        Returns
        -------
        list[SDR]
            One consensus SDR per step.
        """
        rng = np.random.default_rng(self._config.seed + 1)
        outputs: list[SDR] = []
        for i in range(self._config.max_steps):
            if input_stream is not None and i < len(input_stream):
                inp = input_stream[i]
            else:
                inp = rng.standard_normal(self._config.n_input_dim)
            outputs.append(self.step(inp))
        logger.info("Run complete: %d steps", self._config.max_steps)
        return outputs

    def get_reliability_history(self) -> list[dict[str, float]]:
        """Return the per-step trust score snapshots collected during :meth:`run`."""
        return list(self._reliability_history)
'''

# ---------------------------------------------------------------------------
# configs/baseline.yaml
# ---------------------------------------------------------------------------
FILES["configs/baseline.yaml"] = """\
n_units: 4
n_input_dim: 256
max_steps: 100
seed: 42

cortical:
  n_columns: 2048
  sparsity: 0.02
  learning_rate: 0.1

thalamic:
  aggregation: or

trn:
  entropy_threshold: 0.5

reliability:
  initial_score: 0.5
  alpha: 0.1
  min_score: 0.01
  max_score: 1.0

consensus:
  method: weighted_vote
"""

# ---------------------------------------------------------------------------
# experiments/run_baseline.py
# ---------------------------------------------------------------------------
FILES["experiments/run_baseline.py"] = '''\
"""Baseline experiment: run HALO pipeline with the default configuration.

Usage
-----
    python experiments/run_baseline.py
"""

import logging
import sys
from pathlib import Path

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from halo.config.loader import load_config
from halo.orchestration.pipeline import HALOPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def main() -> None:
    config_path = Path(__file__).parent.parent / "configs" / "baseline.yaml"
    config = load_config(config_path)

    logger.info("Starting HALO baseline experiment")
    pipeline = HALOPipeline(config)
    pipeline.run()

    scores = pipeline._reliability.all_scores()
    logger.info("Final reliability scores:")
    for uid, score in sorted(scores.items()):
        logger.info("  %s: %.4f", uid, score)

    logger.info("Total steps completed: %d", config.max_steps)


if __name__ == "__main__":
    main()
'''

# ---------------------------------------------------------------------------
# tests/__init__.py
# ---------------------------------------------------------------------------
FILES["tests/__init__.py"] = ""

# ---------------------------------------------------------------------------
# tests/core/__init__.py
# ---------------------------------------------------------------------------
FILES["tests/core/__init__.py"] = ""

# ---------------------------------------------------------------------------
# tests/core/test_sdr.py
# ---------------------------------------------------------------------------
FILES["tests/core/test_sdr.py"] = '''\
"""Tests for halo.core.sdr.SDR."""

import numpy as np
import pytest

from halo.core.sdr import SDR


def _make_sdr(active: list[int], n: int = 20) -> SDR:
    return SDR.from_indices(np.array(active), n=n, unit_id="test", timestamp=0)


def test_sparsity_invariant() -> None:
    """SDR.sparsity must equal active_count / n."""
    sdr = _make_sdr([0, 1, 2, 3], n=20)
    assert sdr.sparsity == pytest.approx(4 / 20)


def test_overlap_identical() -> None:
    """Overlap of an SDR with itself equals its active count."""
    sdr = _make_sdr([0, 5, 10, 15], n=20)
    assert sdr.overlap(sdr) == 4


def test_overlap_disjoint() -> None:
    """Overlap of two completely disjoint SDRs is 0."""
    a = _make_sdr([0, 1, 2], n=10)
    b = _make_sdr([7, 8, 9], n=10)
    assert a.overlap(b) == 0


def test_from_indices_roundtrip() -> None:
    """SDR.from_indices followed by active_indices reproduces the input."""
    indices = np.array([3, 7, 11, 15])
    sdr = SDR.from_indices(indices, n=20, unit_id="t", timestamp=1)
    recovered = np.sort(sdr.active_indices)
    np.testing.assert_array_equal(recovered, np.sort(indices))


def test_union() -> None:
    """Union SDR has all bits from both operands active."""
    a = _make_sdr([0, 1, 2], n=10)
    b = _make_sdr([3, 4, 5], n=10)
    u = a.union(b)
    expected = np.array([0, 1, 2, 3, 4, 5])
    np.testing.assert_array_equal(np.sort(u.active_indices), expected)


def test_union_overlap() -> None:
    """Union of partially overlapping SDRs deduplicated correctly."""
    a = _make_sdr([0, 1, 2], n=10)
    b = _make_sdr([2, 3, 4], n=10)
    u = a.union(b)
    # bit 2 appears in both — should appear once
    assert int(u.bits.sum()) == 5
    np.testing.assert_array_equal(np.sort(u.active_indices), [0, 1, 2, 3, 4])


def test_n_property() -> None:
    sdr = SDR.empty(100, "u", 0)
    assert sdr.n == 100


def test_empty_sdr_sparsity() -> None:
    sdr = SDR.empty(50, "u", 0)
    assert sdr.sparsity == 0.0


def test_overlap_length_mismatch_raises() -> None:
    a = SDR.empty(10, "a", 0)
    b = SDR.empty(20, "b", 0)
    with pytest.raises(ValueError):
        a.overlap(b)
'''

# ---------------------------------------------------------------------------
# tests/layers/__init__.py
# ---------------------------------------------------------------------------
FILES["tests/layers/__init__.py"] = ""

# ---------------------------------------------------------------------------
# tests/layers/test_trn.py
# ---------------------------------------------------------------------------
FILES["tests/layers/test_trn.py"] = '''\
"""Tests for halo.layers.trn.TRNGatingLayer."""

import numpy as np
import pytest

from halo.config.schema import TRNConfig
from halo.core.sdr import SDR
from halo.layers.trn import TRNGatingLayer


def _constant_sdr(value: bool, n: int = 100, uid: str = "u", ts: int = 0) -> SDR:
    bits = np.full(n, value, dtype=bool)
    return SDR(bits=bits, unit_id=uid, timestamp=ts)


def _sdr_from_seed(seed: int, n: int = 100, sparsity: float = 0.1, ts: int = 0) -> SDR:
    rng = np.random.default_rng(seed)
    bits = rng.random(n) < sparsity
    return SDR(bits=bits, unit_id=f"u_{seed}", timestamp=ts)


def test_no_gating_low_entropy() -> None:
    """Identical SDRs produce zero entropy → all SDRs pass through unchanged."""
    cfg = TRNConfig(entropy_threshold=0.5)
    layer = TRNGatingLayer(cfg)

    # All identical SDRs → entropy = 0
    sdr = _constant_sdr(False, n=100, uid="u0")
    sdrs = [
        SDR(bits=sdr.bits.copy(), unit_id=f"u{i}", timestamp=0) for i in range(4)
    ]
    # Make them non-trivially identical (some active bits)
    for s in sdrs:
        s.bits[:10] = True

    result = layer.process(sdrs)
    assert len(result) == len(sdrs)
    for original, gated in zip(sdrs, result):
        np.testing.assert_array_equal(original.bits, gated.bits)


def test_gating_high_entropy() -> None:
    """Highly varied SDRs → high entropy → at least one SDR suppressed."""
    cfg = TRNConfig(entropy_threshold=0.01)  # very low threshold to ensure gating
    layer = TRNGatingLayer(cfg)

    # Create SDRs with varying sparsity so some are below mean
    sdrs = []
    rng = np.random.default_rng(0)
    for i in range(8):
        bits = rng.random(200) < (0.05 * (i + 1))
        sdrs.append(SDR(bits=bits, unit_id=f"u{i}", timestamp=0))

    result = layer.process(sdrs)
    assert len(result) == len(sdrs)

    # At least one SDR should be suppressed (all-zero)
    suppressed_count = sum(1 for s in result if int(s.bits.sum()) == 0)
    assert suppressed_count >= 1


def test_empty_input_returns_empty() -> None:
    cfg = TRNConfig(entropy_threshold=0.5)
    layer = TRNGatingLayer(cfg)
    assert layer.process([]) == []


def test_single_sdr_passes_through() -> None:
    """A single SDR cannot have high cross-population entropy; it always passes."""
    cfg = TRNConfig(entropy_threshold=0.5)
    layer = TRNGatingLayer(cfg)
    bits = np.zeros(50, dtype=bool)
    bits[:5] = True
    sdr = SDR(bits=bits, unit_id="solo", timestamp=0)
    result = layer.process([sdr])
    assert len(result) == 1
    np.testing.assert_array_equal(result[0].bits, sdr.bits)
'''

# ---------------------------------------------------------------------------
# tests/reliability/__init__.py
# ---------------------------------------------------------------------------
FILES["tests/reliability/__init__.py"] = ""

# ---------------------------------------------------------------------------
# tests/reliability/test_reliability.py
# ---------------------------------------------------------------------------
FILES["tests/reliability/test_reliability.py"] = '''\
"""Tests for halo.reliability.module.ReliabilityModule."""

import pytest

from halo.config.schema import ReliabilityConfig
from halo.reliability.module import ReliabilityModule


def _make_module(initial: float = 0.5) -> ReliabilityModule:
    cfg = ReliabilityConfig(
        initial_score=initial,
        alpha=0.1,
        min_score=0.01,
        max_score=1.0,
    )
    return ReliabilityModule(unit_ids=["unit_0", "unit_1"], config=cfg)


def test_positive_signal_increases_score() -> None:
    mod = _make_module()
    before = mod.get_score("unit_0")
    mod.update("unit_0", signal=1.0)
    assert mod.get_score("unit_0") > before


def test_negative_signal_decreases_score() -> None:
    mod = _make_module()
    before = mod.get_score("unit_0")
    mod.update("unit_0", signal=-1.0)
    assert mod.get_score("unit_0") < before


def test_score_clamped_to_max() -> None:
    mod = _make_module(initial=0.99)
    for _ in range(20):
        mod.update("unit_0", signal=1.0)
    assert mod.get_score("unit_0") <= 1.0


def test_score_clamped_to_min() -> None:
    mod = _make_module(initial=0.02)
    for _ in range(20):
        mod.update("unit_0", signal=-1.0)
    assert mod.get_score("unit_0") >= 0.01


def test_all_scores_returns_all_units() -> None:
    mod = _make_module()
    scores = mod.all_scores()
    assert set(scores.keys()) == {"unit_0", "unit_1"}


def test_all_scores_is_copy() -> None:
    """Mutating the returned dict must not affect internal state."""
    mod = _make_module()
    scores = mod.all_scores()
    scores["unit_0"] = 0.0
    assert mod.get_score("unit_0") == pytest.approx(0.5)


def test_unknown_unit_raises() -> None:
    mod = _make_module()
    with pytest.raises(KeyError):
        mod.get_score("nonexistent")
'''

# ---------------------------------------------------------------------------
# tests/consensus/__init__.py
# ---------------------------------------------------------------------------
FILES["tests/consensus/__init__.py"] = ""

# ---------------------------------------------------------------------------
# tests/consensus/test_consensus.py
# ---------------------------------------------------------------------------
FILES["tests/consensus/test_consensus.py"] = '''\
"""Tests for halo.consensus.engine.ConsensusEngine."""

import numpy as np
import pytest

from halo.config.schema import ConsensusConfig
from halo.consensus.engine import ConsensusEngine
from halo.core.sdr import SDR


def _make_engine() -> ConsensusEngine:
    return ConsensusEngine(ConsensusConfig(method="weighted_vote"))


def _sdr(active: list[int], n: int = 20, uid: str = "u") -> SDR:
    return SDR.from_indices(np.array(active), n=n, unit_id=uid, timestamp=0)


def test_unanimous_vote() -> None:
    """When all units produce the same SDR with equal weights, output == input."""
    engine = _make_engine()
    active = [0, 1, 2, 5, 10]
    sdrs = [_sdr(active, uid=f"unit_{i}") for i in range(4)]
    weights = {f"unit_{i}": 1.0 for i in range(4)}

    result = engine.aggregate(sdrs, weights)
    np.testing.assert_array_equal(result.bits, sdrs[0].bits)


def test_weighted_dominant_unit() -> None:
    """One unit with weight 1.0, others with 0.0 → output == that unit\'s SDR."""
    engine = _make_engine()
    dominant = _sdr([3, 7, 11], n=20, uid="dominant")
    others = [_sdr([0, 1, 2], n=20, uid=f"other_{i}") for i in range(3)]

    sdrs = [dominant] + others
    weights = {"dominant": 1.0, "other_0": 0.0, "other_1": 0.0, "other_2": 0.0}

    result = engine.aggregate(sdrs, weights)
    np.testing.assert_array_equal(result.bits, dominant.bits)


def test_consensus_unit_id() -> None:
    engine = _make_engine()
    sdrs = [_sdr([0, 1], uid=f"u{i}") for i in range(2)]
    weights = {f"u{i}": 1.0 for i in range(2)}
    result = engine.aggregate(sdrs, weights)
    assert result.unit_id == "consensus"


def test_zero_weights_produce_empty_sdr() -> None:
    """All-zero weights → no bits exceed 0.5 → all-zero consensus."""
    engine = _make_engine()
    sdrs = [_sdr([0, 1, 2], uid=f"u{i}") for i in range(3)]
    weights = {f"u{i}": 0.0 for i in range(3)}
    result = engine.aggregate(sdrs, weights)
    assert int(result.bits.sum()) == 0


def test_empty_sdrs_raises() -> None:
    engine = _make_engine()
    with pytest.raises(ValueError):
        engine.aggregate([], {})
'''

# ---------------------------------------------------------------------------
# Write all files
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Setting up HALO project under: {ROOT}")
    for rel, content in FILES.items():
        write(rel, content)
    print(f"\nDone. {len(FILES)} files written.")
    print("\nNext steps:")
    print("  pip install -e '.[dev]'")
    print("  pytest tests/")
    print("  python experiments/run_baseline.py")
