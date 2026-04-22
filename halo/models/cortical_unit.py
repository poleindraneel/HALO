"""CorticalUnit: HTM-inspired spatial pooler with permanence-based overlap scoring.

Implements the HTM Spatial Pooler algorithm (Hawkins et al. 2011) as the
encoding and learning core of each cortical unit in HALO.

Key differences from the previous placeholder:
- Overlap is computed by counting *connected* synapses active on the input,
  not by a dot product.
- Learning updates synapse *permanences* (not weight matrices) via a local
  increment/decrement rule (AdaptSynapses).
- Homeostatic boosting ensures all columns remain competitive over time.

References
----------
Hawkins J. et al. (2011) Hierarchical Temporal Memory — Cortical Learning
    Algorithm white paper. Numenta.
NeoCortexAPI (Dobric): SpatialPooler.cs — CalculateOverlap, AdaptSynapses,
    UpdateBoostFactors, BoostColsWithLowOverlap.
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
