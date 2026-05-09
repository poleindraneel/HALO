"""Heterarchical (non-hierarchical peer-to-peer) lateral mixing layer.

Units at the same processing level influence each other via apical-dendrite-
inspired learned lateral connections.  Each directed edge (v → u) maintains
a sparse weight matrix W[v][u] of shape (n_columns, n_columns).

Forward pass
------------
For each unit u, the lateral bias from connected peers is:

    bias[u] = Σ_{v → u} W[v][u] @ sdr_v.bits

This bias is added to u's feedforward overlap scores *before* winner-take-all
inhibition, but *only* for columns that already have feedforward support
(apical input modulates, it does not drive — Hawkins & Ahmad 2016).

The forward pass operates with a one-step lag: biases at step t are computed
from SDRs at step t-1.  This is biologically grounded (axonal conduction delay).

Learning rule (Hebbian)
-----------------------
After each step, for each directed edge (v → u):

    W[v][u][i, j] += lateral_lr   if col_i active in u AND col_j active in v
    W[v][u][i, j] -= lateral_decay  otherwise (within potential pool only)
    W[v][u] clipped to [0, 1]

References
----------
Apical modulation: Hawkins & Ahmad 2016 "Why Neurons Have Thousands of Synapses".
NAA lateral weights: NeoCortexAPI NAA/NeuralAssociationsAlgorithm.cs.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

from halo.config.schema import HeterarchicalConfig
from halo.core.base import LayerBase
from halo.core.sdr import SDR

logger = logging.getLogger(__name__)

__all__ = ["HeterarchicalLayer"]


class HeterarchicalLayer(LayerBase):
    """Learned lateral connections between peer cortical units.

    Parameters
    ----------
    n_columns:
        Number of minicolumns per cortical unit (must match CorticalUnit).
    config:
        :class:`~halo.config.schema.HeterarchicalConfig` with learning rates
        and initial sparsity.
    rng:
        NumPy random generator for reproducible weight initialisation.

    Usage
    -----
    >>> layer = HeterarchicalLayer(n_columns=2048, config=cfg, rng=rng)
    >>> layer.register_unit("unit_0")
    >>> layer.register_unit("unit_1")
    >>> layer.add_connection("unit_0", "unit_1")
    >>> biases = layer.compute_biases()           # zero on first step
    >>> # ... encode with biases, collect sdrs ...
    >>> layer.update_sdrs(sdrs)                   # cache for next step
    >>> layer.learn(sdrs)                         # Hebbian weight update
    """

    def __init__(
        self,
        n_columns: int,
        config: HeterarchicalConfig,
        rng: np.random.Generator,
    ) -> None:
        self._n = n_columns
        self._config = config
        self._rng = rng

        # Ordered unit list for consistent iteration
        self._unit_ids: list[str] = []

        # adjacency: from_id → list[to_id]
        self._adjacency: dict[str, list[str]] = defaultdict(list)

        # reverse adjacency: to_id → list[from_id]  (populated eagerly in add_connection)
        self._reverse: dict[str, list[str]] = defaultdict(list)

        # Weight matrices: _weights[from_id][to_id] shape (n, n) float32
        self._weights: dict[str, dict[str, np.ndarray]] = {}

        # Sparse potential pool masks: _masks[from_id][to_id] shape (n, n) bool
        self._masks: dict[str, dict[str, np.ndarray]] = {}

        # Cached SDRs from previous step for bias computation
        self._prev_sdrs: dict[str, SDR] = {}

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def register_unit(self, unit_id: str) -> None:
        """Register *unit_id* as a node in the heterarchical graph."""
        if unit_id not in self._weights:
            self._unit_ids.append(unit_id)
            self._weights[unit_id] = {}
            self._masks[unit_id] = {}
        logger.debug("Registered unit %s", unit_id)

    def add_connection(self, from_id: str, to_id: str) -> None:
        """Add a directed lateral connection from *from_id* to *to_id*.

        Allocates a sparse (n_columns × n_columns) weight matrix for this edge,
        with initial weights drawn from Uniform(0, 0.1) within the potential pool.

        Parameters
        ----------
        from_id:
            Source unit — its SDR will modulate *to_id*.
        to_id:
            Target unit — receives lateral bias from *from_id*.

        Raises
        ------
        ValueError
            If either *from_id* or *to_id* has not been registered via
            :meth:`register_unit`.
        """
        if from_id not in self._weights:
            raise ValueError(
                f"Unit '{from_id}' is not registered. Call register_unit('{from_id}') first."
            )
        if to_id not in self._weights:
            raise ValueError(
                f"Unit '{to_id}' is not registered. Call register_unit('{to_id}') first."
            )

        self._adjacency[from_id].append(to_id)
        self._reverse[to_id].append(from_id)

        # Sparse potential pool: each (to_col, from_col) pair connected with
        # probability lateral_sparsity.
        # TODO: replace dense boolean mask + float32 weight matrix with a CSR
        # (scipy.sparse) representation to reduce memory from O(n²) to O(sparsity·n²).
        # At n_columns=2048 and 12 all-to-all edges the dense layout is ~192 MB.
        mask = self._rng.random((self._n, self._n)) < self._config.lateral_sparsity
        self._masks[from_id][to_id] = mask

        # Initialise weights: small random values within pool, zero outside
        w = np.zeros((self._n, self._n), dtype=np.float32)
        n_synapses = int(mask.sum())
        if n_synapses > 0:
            w[mask] = self._rng.uniform(0.0, 0.1, n_synapses).astype(np.float32)
        self._weights[from_id][to_id] = w

        logger.debug(
            "Lateral connection %s → %s: %d synapses (sparsity=%.3f)",
            from_id, to_id, n_synapses, self._config.lateral_sparsity,
        )

    # ------------------------------------------------------------------
    # LayerBase interface
    # ------------------------------------------------------------------

    def process(self, inputs: list[SDR]) -> list[SDR]:
        """Pass inputs through unchanged.

        Lateral modulation is applied *before* this layer is called, inside
        each :meth:`~halo.models.cortical_unit.CorticalUnit.encode` call via
        the ``lateral_bias`` parameter.  This method satisfies the
        :class:`~halo.core.base.LayerBase` ABC.
        """
        return list(inputs)

    def reset(self) -> None:
        """Clear transient per-step state (cached SDRs).

        Learned weight matrices are **preserved** — use :meth:`reset_weights`
        for a full reinitialisation.
        """
        self._prev_sdrs.clear()
        logger.debug("HeterarchicalLayer: transient state cleared (weights preserved)")

    # ------------------------------------------------------------------
    # Lateral bias computation (forward pass)
    # ------------------------------------------------------------------

    def compute_biases(self) -> dict[str, np.ndarray]:
        """Compute lateral bias vectors from the previous step's SDRs.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping from unit_id to a float32 bias vector of length n_columns.
            Units with no lateral inputs (or on the first step) receive
            all-zero bias vectors.
        """
        biases: dict[str, np.ndarray] = {
            uid: np.zeros(self._n, dtype=np.float32) for uid in self._unit_ids
        }

        for to_id, from_ids in self._reverse.items():
            if to_id not in biases:
                continue
            for from_id in from_ids:
                if from_id not in self._prev_sdrs:
                    continue  # no SDR cached yet (first step)
                active = np.flatnonzero(self._prev_sdrs[from_id].bits)
                if active.size:
                    # Sum only columns corresponding to active source bits.
                    # Equivalent to W @ sdr_bits but avoids O(n²) dense matmul.
                    biases[to_id] += self._weights[from_id][to_id][:, active].sum(axis=1)

        return biases

    def update_sdrs(self, sdrs: list[SDR]) -> None:
        """Cache *sdrs* for use in :meth:`compute_biases` on the next step.

        Call this after encoding, before :meth:`learn`.
        """
        for sdr in sdrs:
            self._prev_sdrs[sdr.unit_id] = sdr

    # ------------------------------------------------------------------
    # Hebbian weight update (learning)
    # ------------------------------------------------------------------

    def learn(self, sdrs: list[SDR]) -> None:
        """Update lateral weights using Hebbian co-activity.

        For each directed edge (v → u):

        - Columns co-active in *both* u and v → weight increment within pool.
        - All other pool synapses → weight decay.
        - Weights clipped to [0, 1].

        Only synapses within the potential pool mask are ever updated,
        keeping the effective weight matrix sparse throughout training.

        Parameters
        ----------
        sdrs:
            Current-step SDRs, one per unit.
        """
        sdr_map: dict[str, SDR] = {sdr.unit_id: sdr for sdr in sdrs}

        for from_id, to_ids in self._adjacency.items():
            if from_id not in sdr_map:
                continue
            bits_from: np.ndarray = sdr_map[from_id].bits  # (n,) bool

            for to_id in to_ids:
                if to_id not in sdr_map:
                    continue
                bits_to: np.ndarray = sdr_map[to_id].bits  # (n,) bool

                mask = self._masks[from_id][to_id]   # (n, n) bool
                w = self._weights[from_id][to_id]     # (n, n) float32

                active_from = np.flatnonzero(bits_from)
                active_to = np.flatnonzero(bits_to)

                # Decay all in-pool synapses first
                w[mask] -= self._config.lateral_decay

                # Potentiate co-active in-pool pairs.
                # Using np.ix_ over active indices avoids the O(n²) outer product;
                # at 2% sparsity this is ~40×40 = 1 600 pairs vs 2048²= 4 M.
                if active_from.size and active_to.size:
                    coactive_ix = np.ix_(active_to, active_from)
                    # Undo decay and add lr only for in-pool co-active synapses
                    w[coactive_ix] += (
                        self._config.lateral_lr + self._config.lateral_decay
                    ) * mask[coactive_ix]

                np.clip(w, 0.0, 1.0, out=w)

        logger.debug("HeterarchicalLayer: Hebbian weight update complete")

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def reset_weights(self) -> None:
        """Full reinitialisation: clears learned weights AND transient state."""
        for from_id in self._weights:
            for to_id, mask in self._masks[from_id].items():
                w = np.zeros((self._n, self._n), dtype=np.float32)
                n_synapses = int(mask.sum())
                if n_synapses > 0:
                    w[mask] = self._rng.uniform(0.0, 0.1, n_synapses).astype(np.float32)
                self._weights[from_id][to_id] = w
        self._prev_sdrs.clear()
        logger.debug("HeterarchicalLayer: full weight reset")

    def weight_matrix(self, from_id: str, to_id: str) -> np.ndarray:
        """Return the weight matrix for edge *from_id* → *to_id*.

        Returns a **direct reference** to the internal array — modifications
        will affect the layer's state.  Callers that need a stable snapshot
        should call ``.copy()`` on the result.
        """
        return self._weights[from_id][to_id]

    def potential_mask(self, from_id: str, to_id: str) -> np.ndarray:
        """Return the boolean potential pool mask for edge *from_id* → *to_id*."""
        return self._masks[from_id][to_id]

