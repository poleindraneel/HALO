"""CorticalUnit: HTM Spatial Pooler + Temporal Memory.

Implements the HTM Spatial Pooler (Hawkins et al. 2011) and Temporal Memory
(Hawkins et al. 2011, NeoCortexAPI TemporalMemory.cs) as the encoding and
learning core of each cortical unit in HALO.

Spatial Pooler (encode / learn):
- Overlap is computed by counting *connected* synapses active on the input.
- AdaptSynapses updates permanences with a local increment/decrement rule.
- Homeostatic boosting keeps all columns competitive over time.

Temporal Memory (temporal_step / learn):
- Each minicolumn contains ``cells_per_column`` cells.
- Cells maintain distal dendrite segments with synapse permanences to other cells.
- Predicted columns activate only predicted cells; unpredicted columns burst.
- AdaptSegments reinforces segments that predicted correctly and grows new synapses.

References
----------
Hawkins J. et al. (2011) Hierarchical Temporal Memory — Cortical Learning
    Algorithm white paper. Numenta.
NeoCortexAPI (Dobric): SpatialPooler.cs, TemporalMemory.cs.
"""

from __future__ import annotations

import logging

import numpy as np

from halo.config.schema import CorticalConfig
from halo.core.base import CorticalUnitBase
from halo.core.sdr import SDR

logger = logging.getLogger(__name__)

_SYNAPSE_EPSILON: float = 1e-5  # synapses with permanence below this are pruned

__all__ = ["CorticalUnit"]


class CorticalUnit(CorticalUnitBase):
    """HTM Spatial Pooler cortical unit.

    Each column maintains a permanence value for every input in its potential
    pool.  A synapse is *connected* when its permanence exceeds
    ``syn_perm_connected``.  The overlap score of a column is the count of
    its connected synapses whose corresponding input bit is active.

    Parameters
    ----------
    unit_id:
        Unique identifier string, e.g. ``"unit_0"``.
    config:
        Fully populated :class:`~halo.config.schema.CorticalConfig`.
    rng:
        NumPy random generator for reproducible initialisation.
    input_dim:
        Dimensionality of the binary input vector.
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
        self._k: int = max(1, int(config.sparsity * config.n_columns))

        if not config.global_inhibition or config.local_area_density != 0.0:
            logger.warning(
                "CorticalUnit '%s': local inhibition (global_inhibition=%s, "
                "local_area_density=%s) is not yet implemented; "
                "falling back to sparsity-based global winner selection.",
                unit_id,
                config.global_inhibition,
                config.local_area_density,
            )
        self._init_state()

    # ------------------------------------------------------------------
    # CorticalUnitBase interface
    # ------------------------------------------------------------------

    @property
    def unit_id(self) -> str:
        return self._unit_id

    def encode(self, input_data: np.ndarray) -> SDR:
        """Compute overlap scores and return the winner-take-all SDR.

        Algorithm
        ---------
        1. Build connected mask: permanence >= syn_perm_connected (within pool).
        2. Overlap = count of connected synapses active on *input_data*.
        3. Zero columns below stimulus_threshold.
        4. Multiply by boost factors (homeostatic plasticity).
        5. Top-k columns by boosted overlap → SDR.

        Parameters
        ----------
        input_data:
            Boolean or float array of shape ``(input_dim,)``.  Values > 0
            are treated as active input bits.

        Returns
        -------
        SDR
            Sparse encoding with ``sparsity * n_columns`` active bits.
        """
        input_bool = input_data > 0

        # Step 1 — connected synapses within potential pool
        connected: np.ndarray = (
            self._permanences >= self._config.syn_perm_connected
        ) & self._potential_pool  # (n_columns, input_dim) bool

        # Step 2 — overlap: count of connected + active input bits per column.
        # Cast to int32 so matmul sums counts, not saturated booleans.
        overlaps: np.ndarray = connected @ input_bool.astype(np.int32)  # (n_columns,) int

        # Step 3 — stimulus threshold
        overlaps = np.where(
            overlaps >= self._config.stimulus_threshold, overlaps, 0
        ).astype(float)

        # Step 4 — homeostatic boosting
        boosted: np.ndarray = overlaps * self._boost_factors

        # Step 5 — winner-take-all
        bits = self._winner_take_all(boosted, self._k)

        # Update duty cycles (moving average)
        self._update_overlap_duty_cycle(overlaps > 0)
        self._update_active_duty_cycle(bits)

        self._last_input = input_bool   # cache for learn()
        self._step += 1

        logger.debug(
            "%s encode step=%d active=%d mean_overlap=%.2f",
            self._unit_id,
            self._step,
            int(bits.sum()),
            float(overlaps.mean()),
        )
        return SDR(bits=bits, unit_id=self._unit_id, timestamp=self._step)

    def learn(self, sdr: SDR) -> None:
        """AdaptSynapses: permanence update for all active columns.

        For each active column, iterate over its potential pool:
        - If the corresponding input bit was active → increment permanence.
        - Otherwise → decrement permanence.
        - Clip to ``[0, syn_perm_max]``.
        - Trim values below ``syn_perm_trim_threshold`` to 0.

        Also triggers boost factor update every ``update_period`` steps.

        References
        ----------
        NeoCortexAPI SpatialPooler.cs — AdaptSynapses() line 611.
        """
        active_cols = np.where(sdr.bits)[0]
        if active_cols.size == 0:
            return

        # We need the last input to apply the learning rule.
        # encode() must have been called in this step — use cached input.
        if self._last_input is None:
            logger.warning("%s learn() called without prior encode(); skipping", self._unit_id)
            return

        input_bool = self._last_input

        for col in active_cols:
            pool = self._potential_pool[col]          # (input_dim,) bool
            perm = self._permanences[col]             # (input_dim,) float, view

            # Increment where input active in pool, decrement elsewhere in pool
            delta = np.where(
                pool & input_bool,
                self._config.syn_perm_active_inc,
                np.where(pool, -self._config.syn_perm_inactive_dec, 0.0),
            )
            perm += delta
            np.clip(perm, 0.0, self._config.syn_perm_max, out=perm)
            # Trim near-zero permanences (NeoCortexAPI SynPermTrimThreshold)
            perm[perm < self._config.syn_perm_trim_threshold] = 0.0

        # Homeostatic boost update every update_period steps
        if self._step % self._config.update_period == 0:
            self._update_boost_factors()

        # TM learning — AdaptSegments using state cached by temporal_step()
        self._adapt_segments()

        logger.debug(
            "%s learned from SDR with %d active columns",
            self._unit_id,
            len(active_cols),
        )

    def reset(self) -> None:
        """Reinitialise all internal state and the step counter."""
        self._step = 0
        self._init_state()
        logger.debug("%s reset", self._unit_id)

    def temporal_step(self, column_sdr: SDR) -> SDR:
        """Run HTM Temporal Memory on the Spatial Pooler output.

        Two phases (NeoCortexAPI TemporalMemory.cs):

        **Phase 1 — ActivateCells:**
        For each active column from the SP output:

        - *Predicted column* (has cells in ``_predictive_cells``): activate
          only those cells; each becomes a winner.
        - *Bursting column* (no predictive cells): activate all cells; winner
          is the cell with the best matching segment to previous active cells
          (tie-break: fewest segments — deterministic).

        Punishment: columns predicted last step but not active this step have
        their matching segments decremented by ``predicted_segment_decrement``.

        **Phase 2 — ActivateDendrites:**
        Compute active and matching segments from current active cells.
        Update ``_predictive_cells`` for the next step.

        Parameters
        ----------
        column_sdr:
            Column-level SDR from :meth:`encode`.

        Returns
        -------
        SDR
            Cell-level SDR with ``n_columns * cells_per_column`` bits.

        Notes
        -----
        TM learning (AdaptSegments / synapse growth) is deferred to the next
        :meth:`learn` call, which uses ``_winner_cells``,
        ``_learning_seg_for_winner``, and ``_prev_winner_cells`` cached here.

        References
        ----------
        NeoCortexAPI (Dobric): TemporalMemory.cs — ActivateCells,
            ActivateDendrites, GetBestMatchingCell.
        """
        cfg = self._config
        n_cpc = cfg.cells_per_column
        active_cols: set[int] = {int(i) for i in np.where(column_sdr.bits)[0]}

        # ----------------------------------------------------------------
        # Phase 1 — ActivateCells
        # ----------------------------------------------------------------
        active_cells: set[int] = set()
        winner_cells: set[int] = set()
        learning_seg_for_winner: dict[int, int | None] = {}

        for col in active_cols:
            first = col * n_cpc
            col_cells = range(first, first + n_cpc)
            predicted = [c for c in col_cells if c in self._predictive_cells]

            if predicted:
                # Correctly predicted column — activate only predictive cells
                for cell in predicted:
                    active_cells.add(cell)
                    winner_cells.add(cell)
                    learning_seg_for_winner[cell] = self._learning_seg_for_predicted.get(cell)
            else:
                # Bursting column — activate all cells; elect one winner
                for cell in col_cells:
                    active_cells.add(cell)
                winner = self._best_matching_cell(list(col_cells))
                winner_cells.add(winner)
                learning_seg_for_winner[winner] = self._best_matching_seg(winner)

        # Punish incorrectly predicted columns
        if cfg.predicted_segment_decrement > 0.0:
            for col in self._prev_predicted_columns - active_cols:
                first = col * n_cpc
                for cell in range(first, first + n_cpc):
                    for seg_idx, seg in enumerate(self._segments[cell]):
                        if (cell, seg_idx) in self._matching_segments:
                            for pre in list(seg.keys()):
                                if pre in self._prev_active_cells:
                                    new_perm = seg[pre] - cfg.predicted_segment_decrement
                                    if new_perm < _SYNAPSE_EPSILON:
                                        del seg[pre]
                                    else:
                                        seg[pre] = new_perm

        # ----------------------------------------------------------------
        # Phase 2 — ActivateDendrites (predicts next step)
        # ----------------------------------------------------------------
        new_active_segs: set[tuple[int, int]] = set()
        new_matching_segs: set[tuple[int, int]] = set()
        new_predictive_cells: set[int] = set()
        new_learning_seg_for_predicted: dict[int, int] = {}
        new_predicted_cols: set[int] = set()

        for cell_idx, segs in enumerate(self._segments):
            for seg_idx, seg in enumerate(segs):
                connected = sum(
                    1 for pre, perm in seg.items()
                    if pre in active_cells and perm >= cfg.syn_perm_connected
                )
                potential = sum(1 for pre in seg if pre in active_cells)
                if connected >= cfg.activation_threshold:
                    new_active_segs.add((cell_idx, seg_idx))
                    new_predictive_cells.add(cell_idx)
                    if cell_idx not in new_learning_seg_for_predicted:
                        new_learning_seg_for_predicted[cell_idx] = seg_idx
                    new_predicted_cols.add(cell_idx // n_cpc)
                elif potential >= cfg.min_threshold:
                    new_matching_segs.add((cell_idx, seg_idx))

        # Carry forward for learn()
        self._prev_active_cells = self._active_cells
        self._prev_winner_cells = self._winner_cells

        # Update current-step TM state
        self._active_cells = active_cells
        self._winner_cells = winner_cells
        self._learning_seg_for_winner = learning_seg_for_winner
        self._active_segments = new_active_segs
        self._matching_segments = new_matching_segs
        self._predictive_cells = new_predictive_cells
        self._learning_seg_for_predicted = new_learning_seg_for_predicted
        self._prev_predicted_columns = new_predicted_cols

        logger.debug(
            "%s temporal_step: %d active cols → %d active cells, %d predictive",
            self._unit_id,
            len(active_cols),
            len(active_cells),
            len(new_predictive_cells),
        )

        n_total = cfg.n_columns * n_cpc
        bits = np.zeros(n_total, dtype=bool)
        for cell in active_cells:
            bits[cell] = True
        return SDR(bits=bits, unit_id=self._unit_id, timestamp=self._step)

    # ------------------------------------------------------------------
    # Private — initialisation
    # ------------------------------------------------------------------

    def _init_state(self) -> None:
        """Initialise permanences, potential pool, boost factors, duty cycles."""
        cfg = self._config
        n, d = cfg.n_columns, self._input_dim

        # Potential pool — which inputs each column may connect to
        self._potential_pool: np.ndarray = self._init_potential_pool(n, d)

        # Permanences — initialised around syn_perm_connected with small noise
        # Connected inputs start slightly above threshold; others below.
        # Pattern from NeoCortexAPI ConnectAndConfigureInputs().
        noise = self._rng.uniform(-0.1, 0.1, size=(n, d))
        base = np.where(
            self._potential_pool,
            cfg.syn_perm_connected + noise,
            cfg.syn_perm_connected - 0.1 + noise * 0.5,
        )
        self._permanences: np.ndarray = np.clip(base, 0.0, cfg.syn_perm_max).astype(
            np.float32
        )

        # Homeostatic state
        self._boost_factors: np.ndarray = np.ones(n, dtype=np.float32)
        self._overlap_duty_cycles: np.ndarray = np.zeros(n, dtype=np.float32)
        self._active_duty_cycles: np.ndarray = np.zeros(n, dtype=np.float32)

        # Cache last input for learn()
        self._last_input: np.ndarray | None = None

        # ------------------------------------------------------------------
        # Temporal memory state
        # Each cell is addressed by flat index: col * cells_per_column + cell_in_col
        # ------------------------------------------------------------------
        n_total_cells = cfg.n_columns * cfg.cells_per_column
        # _segments[cell] = list of segment dicts; each dict: {pre_cell_idx: permanence}
        self._segments: list[list[dict[int, float]]] = [[] for _ in range(n_total_cells)]
        self._active_cells: set[int] = set()
        self._winner_cells: set[int] = set()
        self._prev_active_cells: set[int] = set()
        self._prev_winner_cells: set[int] = set()
        self._predictive_cells: set[int] = set()
        self._active_segments: set[tuple[int, int]] = set()    # (cell, seg_idx)
        self._matching_segments: set[tuple[int, int]] = set()  # (cell, seg_idx)
        self._prev_predicted_columns: set[int] = set()
        # Maps winner cell → seg_idx to reinforce in learn() (None = grow new segment)
        self._learning_seg_for_winner: dict[int, int | None] = {}
        # Maps predictive cell → seg_idx that caused prediction (used next ActivateCells)
        self._learning_seg_for_predicted: dict[int, int] = {}

    def _init_potential_pool(self, n_columns: int, input_dim: int) -> np.ndarray:
        """Build the boolean potential pool matrix.

        If ``potential_radius == -1`` (global), each column samples
        ``potential_pct`` of all inputs as its potential pool.
        Otherwise each column samples ``potential_pct`` of the inputs within
        its local neighbourhood (radius steps either side of the mapped centre).
        """
        cfg = self._config
        pool = np.zeros((n_columns, input_dim), dtype=bool)

        if cfg.potential_radius == -1:
            # Global connectivity — sample potential_pct of ALL inputs
            n_potential = max(1, int(cfg.potential_pct * input_dim))
            for col in range(n_columns):
                indices = self._rng.choice(input_dim, size=n_potential, replace=False)
                pool[col, indices] = True
        else:
            # Local neighbourhood — centre each column on a mapped input index
            for col in range(n_columns):
                centre = int(col / n_columns * input_dim)
                lo = max(0, centre - cfg.potential_radius)
                hi = min(input_dim, centre + cfg.potential_radius + 1)
                candidates = np.arange(lo, hi)
                n_potential = max(1, int(cfg.potential_pct * len(candidates)))
                indices = self._rng.choice(candidates, size=n_potential, replace=False)
                pool[col, indices] = True

        return pool

    # ------------------------------------------------------------------
    # Private — duty cycle updates
    # ------------------------------------------------------------------

    def _update_overlap_duty_cycle(self, had_overlap: np.ndarray) -> None:
        """Exponential moving average of whether each column had overlap > 0."""
        period = float(self._config.duty_cycle_period)
        self._overlap_duty_cycles = (
            (period - 1.0) * self._overlap_duty_cycles + had_overlap.astype(float)
        ) / period

    def _update_active_duty_cycle(self, active: np.ndarray) -> None:
        """Exponential moving average of column activation."""
        period = float(self._config.duty_cycle_period)
        self._active_duty_cycles = (
            (period - 1.0) * self._active_duty_cycles + active.astype(float)
        ) / period

    # ------------------------------------------------------------------
    # Private — homeostatic plasticity
    # ------------------------------------------------------------------

    def _update_boost_factors(self) -> None:
        """Recompute boost factors and nudge permanences for starved columns.

        References
        ----------
        NeoCortexAPI SpatialPooler.cs — UpdateBoostFactors() line 1150,
        BoostColsWithLowOverlap() line 658.
        """
        cfg = self._config

        # Active duty cycle boosting
        max_adc = float(self._active_duty_cycles.max()) or 1.0
        min_adc = cfg.min_pct_active_duty_cycles * max_adc

        boost = np.where(
            self._active_duty_cycles < min_adc,
            1.0 + (cfg.max_boost - 1.0)
            * np.where(
                min_adc > 0,
                (min_adc - self._active_duty_cycles) / min_adc,
                0.0,
            ),
            1.0,
        )
        self._boost_factors = boost.astype(np.float32)

        # Overlap duty cycle — nudge permanences for starved columns
        max_odc = float(self._overlap_duty_cycles.max()) or 1.0
        min_odc = cfg.min_pct_overlap_duty_cycles * max_odc
        starved = self._overlap_duty_cycles < min_odc
        if starved.any():
            self._permanences[starved] += cfg.syn_perm_below_stimulus_inc
            np.clip(self._permanences, 0.0, cfg.syn_perm_max, out=self._permanences)
            logger.debug(
                "%s boosted permanences for %d starved columns",
                self._unit_id,
                int(starved.sum()),
            )

    # ------------------------------------------------------------------
    # Private — winner-take-all
    # ------------------------------------------------------------------

    def _winner_take_all(self, scores: np.ndarray, k: int) -> np.ndarray:
        """Return boolean mask with True at the top-*k* score positions."""
        indices = np.argpartition(scores, -k)[-k:]
        mask = np.zeros(len(scores), dtype=bool)
        mask[indices] = True
        return mask

    # ------------------------------------------------------------------
    # Private — temporal memory
    # ------------------------------------------------------------------

    def _best_matching_cell(self, col_cells: list[int]) -> int:
        """Return the winner cell for a bursting column.

        Winner = cell whose matching segment has the most synapses to
        ``_prev_active_cells`` with overlap ≥ ``min_threshold``.

        Tie-break / no match: cell with the fewest existing segments
        (deterministic — avoids rng dependency in tests).

        References
        ----------
        NeoCortexAPI TemporalMemory.cs — GetBestMatchingCell().
        """
        best_cell: int | None = None
        best_overlap = -1
        for cell in col_cells:
            for seg in self._segments[cell]:
                overlap = sum(1 for pre in seg if pre in self._prev_active_cells)
                if overlap >= self._config.min_threshold and overlap > best_overlap:
                    best_overlap = overlap
                    best_cell = cell
        if best_cell is None:
            best_cell = min(col_cells, key=lambda c: len(self._segments[c]))
        return best_cell

    def _best_matching_seg(self, cell: int) -> int | None:
        """Return the index of the best matching segment on *cell*, or None.

        Best = segment with the most synapses to ``_prev_active_cells``
        with overlap ≥ ``min_threshold``.
        """
        best_idx: int | None = None
        best_overlap = -1
        for seg_idx, seg in enumerate(self._segments[cell]):
            overlap = sum(1 for pre in seg if pre in self._prev_active_cells)
            if overlap >= self._config.min_threshold and overlap > best_overlap:
                best_overlap = overlap
                best_idx = seg_idx
        return best_idx

    def _adapt_segments(self) -> None:
        """TM AdaptSegments: update permanences for winner cells this step.

        For each winner cell:

        - If a learning segment exists (segment index in
          ``_learning_seg_for_winner``): reinforce synapses to
          ``_prev_winner_cells``, weaken others, prune below
          ``_SYNAPSE_EPSILON``, then grow new synapses.
        - Otherwise: create a new segment and grow synapses from
          ``_prev_winner_cells`` (skipped if empty — first step).

        Empty segments are removed after adapting.

        References
        ----------
        NeoCortexAPI TemporalMemory.cs — AdaptSegment(), GrowSynapses().
        """
        cfg = self._config
        for cell, seg_idx in self._learning_seg_for_winner.items():
            if seg_idx is not None and seg_idx < len(self._segments[cell]):
                seg = self._segments[cell][seg_idx]
                for pre in list(seg.keys()):
                    if pre in self._prev_winner_cells:
                        seg[pre] = min(1.0, seg[pre] + cfg.permanence_increment)
                    else:
                        new_perm = seg[pre] - cfg.permanence_decrement
                        if new_perm < _SYNAPSE_EPSILON:
                            del seg[pre]
                        else:
                            seg[pre] = new_perm
                self._grow_synapses(cell, seg_idx)
            else:
                # No learning segment — create one if we have prior context
                if self._prev_winner_cells:
                    self._segments[cell].append({})
                    self._grow_synapses(cell, len(self._segments[cell]) - 1)

        # Prune empty segments
        for cell in self._winner_cells:
            self._segments[cell] = [s for s in self._segments[cell] if s]

    def _grow_synapses(self, cell: int, seg_idx: int) -> None:
        """Grow new synapses from ``_prev_winner_cells`` onto *seg_idx* of *cell*.

        Only synapses not already present and not self-connections are added.
        New synapses start at ``initial_permanence``.
        Candidates are sorted for deterministic behaviour.
        """
        cfg = self._config
        seg = self._segments[cell][seg_idx]
        existing = set(seg.keys()) | {cell}
        candidates = sorted(self._prev_winner_cells - existing)
        n_to_grow = max(0, cfg.max_new_synapse_count - len(seg))
        for pre in candidates[:n_to_grow]:
            seg[pre] = cfg.initial_permanence
