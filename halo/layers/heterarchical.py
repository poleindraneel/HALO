"""Heterarchical (non-hierarchical peer-to-peer) lateral mixing layer.

Units at the same processing level influence each other via apical-dendrite-
inspired learned lateral connections.  Each directed edge (v → u) maintains
a sparse weight structure with shape ``(n_columns, n_columns)`` semantically
but stored as a CSR-style triple ``(indptr, indices, data)`` so memory and
compute scale with the number of actual synapses, not ``n_columns ** 2``.

Memory comparison (n_columns=2048, lateral_sparsity=0.02, 12 edges)
-------------------------------------------------------------------
- Dense layout (legacy):  ~240 MB  (16 MB weights + 4 MB mask per edge × 12)
- CSR layout (this file): ~8 MB    (≈ 670 KB per edge × 12)

Forward pass
------------
For each unit u, the lateral bias from connected peers is:

    bias[u] = Σ_{v → u} W[v→u] @ sdr_v.bits

This bias is added to u's feedforward overlap scores *before* winner-take-all
inhibition, but *only* for columns that already have feedforward support
(apical input modulates, it does not drive — Hawkins & Ahmad 2016).

The forward pass operates with a one-step lag: biases at step t are computed
from SDRs at step t-1.  This is biologically grounded (axonal conduction delay).

Learning rule (Hebbian)
-----------------------
After each step, for each directed edge (v → u):

- All in-pool synapses decay by ``lateral_decay``.
- Synapses whose ``to`` column is active in u AND ``from`` column is active in v
  receive a potentiation of ``lateral_lr + lateral_decay`` (net +lateral_lr after
  undoing the decay).
- Weights clipped to ``[0, 1]``.

Only synapses within the potential pool are ever updated — out-of-pool
entries simply do not exist in the sparse storage.

CSR layout
----------
Per edge, storage is organised by *destination* column (the row index):

    indptr  : (n_columns + 1,) int32 — CSR row pointers
    indices : (nnz,) int32           — source column for each synapse, sorted within row
    data    : (nnz,) float32         — synaptic weight
    row_idx : (nnz,) int32           — destination column for each synapse
                                       (cached for vectorised bias computation)

``indices`` is sorted within each row so per-row membership checks can be
performed with a boolean lookup array of size ``n_columns``.

References
----------
Apical modulation: Hawkins & Ahmad 2016 "Why Neurons Have Thousands of Synapses".
NAA lateral weights: NeoCortexAPI NAA/NeuralAssociationsAlgorithm.cs.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from halo.config.schema import HeterarchicalConfig
from halo.core.base import LayerBase
from halo.core.sdr import SDR

logger = logging.getLogger(__name__)

__all__ = ["HeterarchicalLayer"]


@dataclass
class _CSREdge:
    """Internal CSR storage for one directed lateral edge.

    Attributes
    ----------
    indptr:
        Row pointer array of length ``n_columns + 1``.  Synapses targeting
        column ``r`` are at positions ``indptr[r]:indptr[r+1]`` in
        :attr:`indices` and :attr:`data`.
    indices:
        Source-column index of each synapse, sorted ascending within each row.
    data:
        Synaptic weight of each synapse, aligned with :attr:`indices`.
    row_idx:
        Destination column index for each synapse (cached for fast
        bias accumulation; equivalent to ``np.repeat(arange(n), diff(indptr))``).
    """

    indptr: np.ndarray   # (n + 1,) int32
    indices: np.ndarray  # (nnz,) int32
    data: np.ndarray     # (nnz,) float32
    row_idx: np.ndarray  # (nnz,) int32

    @property
    def nnz(self) -> int:
        return int(self.data.size)


class HeterarchicalLayer(LayerBase):
    """Learned lateral connections between peer cortical units (CSR-sparse).

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

        # CSR edges: _edges[from_id][to_id] -> _CSREdge
        self._edges: dict[str, dict[str, _CSREdge]] = {}

        # Cached SDRs from previous step for bias computation
        self._prev_sdrs: dict[str, SDR] = {}

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def register_unit(self, unit_id: str) -> None:
        """Register *unit_id* as a node in the heterarchical graph."""
        if unit_id not in self._edges:
            self._unit_ids.append(unit_id)
            self._edges[unit_id] = {}
        logger.debug("Registered unit %s", unit_id)

    def add_connection(self, from_id: str, to_id: str) -> None:
        """Add a directed lateral connection from *from_id* to *to_id*.

        Allocates a CSR-sparse weight structure for this edge.  Each
        (destination, source) pair is sampled into the potential pool
        independently with probability ``lateral_sparsity``.  In-pool
        synapses are initialised from ``Uniform(0, 0.1)``.

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
        if from_id not in self._edges:
            raise ValueError(
                f"Unit '{from_id}' is not registered. "
                f"Call register_unit('{from_id}') first."
            )
        if to_id not in self._edges:
            raise ValueError(
                f"Unit '{to_id}' is not registered. "
                f"Call register_unit('{to_id}') first."
            )

        self._adjacency[from_id].append(to_id)
        self._reverse[to_id].append(from_id)

        edge = self._sample_edge()
        self._edges[from_id][to_id] = edge

        logger.debug(
            "Lateral connection %s → %s: %d synapses (sparsity=%.3f)",
            from_id, to_id, edge.nnz, self._config.lateral_sparsity,
        )

    def _sample_edge(self) -> _CSREdge:
        """Sample a new CSR edge using the configured ``lateral_sparsity``.

        Generates a row-by-row Bernoulli potential pool.  ``indices`` is
        guaranteed sorted within each row (a property of ``np.flatnonzero``).
        """
        n = self._n
        p = self._config.lateral_sparsity
        indptr = np.zeros(n + 1, dtype=np.int32)
        rows_idx: list[np.ndarray] = []
        for r in range(n):
            row_mask = self._rng.random(n) < p
            cols = np.flatnonzero(row_mask).astype(np.int32)
            rows_idx.append(cols)
            indptr[r + 1] = indptr[r] + cols.size

        if rows_idx:
            indices = np.concatenate(rows_idx).astype(np.int32, copy=False)
        else:
            indices = np.empty(0, dtype=np.int32)
        nnz = int(indices.size)
        data = self._rng.uniform(0.0, 0.1, size=nnz).astype(np.float32)
        row_idx = np.repeat(np.arange(n, dtype=np.int32), np.diff(indptr))
        return _CSREdge(indptr=indptr, indices=indices, data=data, row_idx=row_idx)

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

        Learned weights are **preserved** — use :meth:`reset_weights` for a
        full reinitialisation.
        """
        self._prev_sdrs.clear()
        logger.debug("HeterarchicalLayer: transient state cleared (weights preserved)")

    # ------------------------------------------------------------------
    # Lateral bias computation (forward pass)
    # ------------------------------------------------------------------

    def compute_biases(self) -> dict[str, np.ndarray]:
        """Compute lateral bias vectors from the previous step's SDRs.

        Algorithm
        ---------
        For each incoming edge ``(from → to)``, build an ``is_active`` lookup
        of length ``n_columns`` from the source SDR; multiply the edge's
        ``data`` by ``is_active[indices]`` (zeros out inactive sources); then
        sum per destination row using ``np.bincount`` over the cached
        ``row_idx``.  Complexity is ``O(nnz_edge)`` per edge.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping ``unit_id`` → float32 bias vector of length ``n_columns``.
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
                edge = self._edges[from_id][to_id]
                if edge.nnz == 0:
                    continue
                src_bits: np.ndarray = self._prev_sdrs[from_id].bits  # (n,) bool
                if not src_bits.any():
                    continue
                # Boolean lookup: which source columns are active.
                contrib = edge.data * src_bits[edge.indices].astype(np.float32)
                # Row-wise sum via bincount, aligned to destination columns.
                biases[to_id] += np.bincount(
                    edge.row_idx, weights=contrib, minlength=self._n
                ).astype(np.float32, copy=False)

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

        Algorithm (per edge, all vectorised over the CSR arrays):

        1. ``data -= lateral_decay`` (one pass over all in-pool synapses).
        2. For each row ``to`` with ``bits_to[to] == True``, locate synapses
           whose source column is also active and add
           ``lateral_lr + lateral_decay`` (undoing the decay and applying the
           potentiation step).
        3. ``np.clip(data, 0, 1)``.

        Out-of-pool entries do not exist in the sparse store, so the
        "weights stay zero outside the mask" invariant is automatic.

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
                edge = self._edges[from_id][to_id]
                if edge.nnz == 0:
                    continue

                # Step 1 — decay all in-pool synapses
                edge.data -= self._config.lateral_decay

                # Step 2 — potentiate co-active (to, from) synapses
                if bits_to.any() and bits_from.any():
                    delta = self._config.lateral_lr + self._config.lateral_decay
                    active_to = np.flatnonzero(bits_to)
                    for to in active_to:
                        start = int(edge.indptr[to])
                        end = int(edge.indptr[to + 1])
                        if end == start:
                            continue
                        row_cols = edge.indices[start:end]
                        active_mask = bits_from[row_cols]
                        if active_mask.any():
                            edge.data[start:end][active_mask] += delta

                # Step 3 — clip
                np.clip(edge.data, 0.0, 1.0, out=edge.data)

        logger.debug("HeterarchicalLayer: Hebbian weight update complete")

    # ------------------------------------------------------------------
    # Public inspection API
    # ------------------------------------------------------------------

    def n_synapses(self, from_id: str, to_id: str) -> int:
        """Return the number of in-pool synapses on edge *from_id* → *to_id*."""
        return self._edges[from_id][to_id].nnz

    def get_weight(self, from_id: str, to_id: str, to_col: int, from_col: int) -> float:
        """Return the weight of one synapse, or ``0.0`` if outside the pool.

        Uses a binary search within the destination row, so this is
        ``O(log(row_nnz))`` per call.
        """
        edge = self._edges[from_id][to_id]
        start = int(edge.indptr[to_col])
        end = int(edge.indptr[to_col + 1])
        row_cols = edge.indices[start:end]
        pos = np.searchsorted(row_cols, from_col)
        if pos < row_cols.size and row_cols[pos] == from_col:
            return float(edge.data[start + pos])
        return 0.0

    def edge_arrays(
        self, from_id: str, to_id: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the raw CSR arrays for inspection.

        Returns
        -------
        (indptr, indices, data, row_idx)
            All four arrays are **direct references** — modifications affect
            layer state.  Callers needing snapshots should ``.copy()``.
        """
        edge = self._edges[from_id][to_id]
        return edge.indptr, edge.indices, edge.data, edge.row_idx

    def weight_matrix(self, from_id: str, to_id: str) -> np.ndarray:
        """Return a dense ``(n_columns, n_columns)`` materialisation of the edge.

        .. warning::
            This allocates an ``O(n²)`` array on every call and is intended
            for tests/debugging only.  Mutations to the returned array do
            **not** propagate back to the sparse store — use
            :meth:`set_all_weights` or :meth:`set_dense_weights` instead.
        """
        edge = self._edges[from_id][to_id]
        w = np.zeros((self._n, self._n), dtype=np.float32)
        if edge.nnz:
            w[edge.row_idx, edge.indices] = edge.data
        return w

    def potential_mask(self, from_id: str, to_id: str) -> np.ndarray:
        """Return a dense ``(n_columns, n_columns)`` boolean mask of the pool.

        .. warning::
            Materialises an ``O(n²)`` boolean array.  For tests/debugging only.
        """
        edge = self._edges[from_id][to_id]
        mask = np.zeros((self._n, self._n), dtype=bool)
        if edge.nnz:
            mask[edge.row_idx, edge.indices] = True
        return mask

    # ------------------------------------------------------------------
    # Public mutation API
    # ------------------------------------------------------------------

    def set_all_weights(self, from_id: str, to_id: str, value: float) -> None:
        """Set every in-pool synapse on edge *from_id* → *to_id* to *value*.

        Equivalent to ``weight_matrix(from_id, to_id)[mask] = value`` in the
        legacy dense API.  Out-of-pool entries are untouched (they don't exist).
        """
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"value must be in [0, 1], got {value}")
        edge = self._edges[from_id][to_id]
        edge.data.fill(np.float32(value))

    def set_dense_weights(
        self, from_id: str, to_id: str, dense: np.ndarray
    ) -> None:
        """Copy values from a dense ``(n_columns, n_columns)`` array into the
        sparse store, restricted to in-pool entries.

        Out-of-pool entries of *dense* are silently ignored.  This is the
        sparse-aware replacement for ``np.fill_diagonal(weight_matrix(...), v)``
        and similar dense mutations.
        """
        if dense.shape != (self._n, self._n):
            raise ValueError(
                f"dense must have shape ({self._n}, {self._n}), got {dense.shape}"
            )
        edge = self._edges[from_id][to_id]
        if edge.nnz == 0:
            return
        edge.data[:] = dense[edge.row_idx, edge.indices].astype(np.float32, copy=False)
        np.clip(edge.data, 0.0, 1.0, out=edge.data)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def reset_weights(self) -> None:
        """Full reinitialisation: re-sample weights AND clear transient state.

        The potential pool topology is preserved; only the synaptic ``data``
        values are re-sampled from ``Uniform(0, 0.1)``.
        """
        for from_id, edges_to in self._edges.items():
            for to_id, edge in edges_to.items():
                edge.data[:] = self._rng.uniform(0.0, 0.1, size=edge.nnz).astype(
                    np.float32, copy=False
                )
        self._prev_sdrs.clear()
        logger.debug("HeterarchicalLayer: full weight reset")
