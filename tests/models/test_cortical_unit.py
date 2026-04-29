"""Tests for halo.models.cortical_unit.CorticalUnit (issue #14)."""

from __future__ import annotations

import numpy as np
import pytest

from halo.config.schema import CorticalConfig
from halo.models.cortical_unit import CorticalUnit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _config(**overrides) -> CorticalConfig:
    defaults = dict(
        n_columns=128,
        sparsity=0.1,
        potential_radius=-1,
        potential_pct=0.5,
        syn_perm_connected=0.5,
        syn_perm_active_inc=0.1,
        syn_perm_inactive_dec=0.01,
        syn_perm_max=1.0,
        stimulus_threshold=0.0,
        local_area_density=0.02,
        global_inhibition=True,
        max_boost=10.0,
        duty_cycle_period=1000,
        min_pct_overlap_duty_cycles=0.001,
        min_pct_active_duty_cycles=0.001,
        update_period=50,
        learning_rate=0.1,
    )
    defaults.update(overrides)
    return CorticalConfig(**defaults)


def _unit(config: CorticalConfig | None = None, input_dim: int = 64) -> CorticalUnit:
    cfg = config or _config()
    rng = np.random.default_rng(42)
    return CorticalUnit(unit_id="test_unit", config=cfg, rng=rng, input_dim=input_dim)


def _random_input(input_dim: int = 64, sparsity: float = 0.1, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bits = rng.random(input_dim) < sparsity
    return bits.astype(float)


# ---------------------------------------------------------------------------
# encode() — output correctness
# ---------------------------------------------------------------------------

def test_encode_returns_sdr_with_correct_sparsity() -> None:
    """Active bit count must equal floor(sparsity * n_columns)."""
    unit = _unit()
    inp = _random_input()
    sdr = unit.encode(inp)
    expected_k = max(1, int(unit._config.sparsity * unit._config.n_columns))
    assert int(sdr.bits.sum()) == expected_k


def test_encode_sdr_unit_id_matches() -> None:
    unit = _unit()
    sdr = unit.encode(_random_input())
    assert sdr.unit_id == "test_unit"


def test_encode_timestamp_increments() -> None:
    unit = _unit()
    s1 = unit.encode(_random_input(seed=0))
    s2 = unit.encode(_random_input(seed=1))
    assert s2.timestamp == s1.timestamp + 1


def test_connected_synapses_only_count() -> None:
    """Only synapses with permanence >= syn_perm_connected contribute to overlap.

    Setup: col 0 and col 1 both have potential connections to input 0.
    When col 0's permanence is above the connected threshold and col 1's is
    below, only col 0 accumulates overlap → col 0 must win WTA (k=1).
    Swapping permanences reverses the winner, proving the threshold matters.
    """
    cfg = _config(n_columns=10, sparsity=0.1, stimulus_threshold=0.0)  # k=1
    rng = np.random.default_rng(0)
    unit = CorticalUnit(unit_id="u", config=cfg, rng=rng, input_dim=10)

    unit._potential_pool[:] = False
    unit._permanences[:] = 0.0
    # Both cols compete for input 0
    unit._potential_pool[0, 0] = True
    unit._potential_pool[1, 0] = True

    inp = np.zeros(10, dtype=float)
    inp[0] = 1.0

    # Col 0 connected, col 1 not → col 0 gets overlap 1, col 1 gets 0 → col 0 wins
    unit._permanences[0, 0] = cfg.syn_perm_connected + 0.1
    unit._permanences[1, 0] = cfg.syn_perm_connected - 0.1
    sdr = unit.encode(inp)
    assert sdr.bits[0], "Col 0 should win: its synapse is connected"
    assert not sdr.bits[1], "Col 1 should not win: its synapse is below threshold"

    # Swap: col 1 connected, col 0 not → col 1 wins
    unit._permanences[0, 0] = cfg.syn_perm_connected - 0.1
    unit._permanences[1, 0] = cfg.syn_perm_connected + 0.1
    sdr = unit.encode(inp)
    assert sdr.bits[1], "Col 1 should win after permanences are swapped"
    assert not sdr.bits[0], "Col 0 should not win: its synapse is now below threshold"


def test_encode_stimulus_threshold_zeros_low_overlap() -> None:
    """Columns with overlap < stimulus_threshold are excluded from WTA winners.

    Setup: col 0 has 3 connected synapses on active inputs (overlap=3),
    col 1 has 1 connected synapse on an active input (overlap=1).
    With stimulus_threshold=3, col 1's overlap is zeroed out and col 0
    must be the sole winner (k=1).
    """
    cfg = _config(n_columns=10, sparsity=0.1, stimulus_threshold=3.0)  # k=1
    rng = np.random.default_rng(0)
    unit = CorticalUnit(unit_id="u", config=cfg, rng=rng, input_dim=10)

    unit._potential_pool[:] = False
    unit._permanences[:] = 0.0

    # Col 0: 3 connected synapses → overlap = 3 >= threshold
    unit._potential_pool[0, :3] = True
    unit._permanences[0, :3] = cfg.syn_perm_connected + 0.1

    # Col 1: 1 connected synapse → overlap = 1 < threshold (will be zeroed)
    unit._potential_pool[1, 3] = True
    unit._permanences[1, 3] = cfg.syn_perm_connected + 0.1

    inp = np.zeros(10, dtype=float)
    inp[:4] = 1.0  # inputs 0–3 all active

    sdr = unit.encode(inp)
    assert sdr.bits[0], "Col 0 (overlap 3 >= threshold 3) must win"
    assert not sdr.bits[1], "Col 1 (overlap 1 < threshold 3) must not win"


def test_potential_pool_global_gives_each_column_a_potential_connection() -> None:
    """With potential_radius=-1, every column has at least one potential connection."""
    unit = _unit()
    # Each column should have at least one potential connection
    assert unit._potential_pool.any(axis=1).all()


def test_boost_factors_init_to_one() -> None:
    unit = _unit()
    np.testing.assert_array_equal(unit._boost_factors, np.ones(unit._config.n_columns))


# ---------------------------------------------------------------------------
# learn() — permanence updates
# ---------------------------------------------------------------------------

def test_learn_increases_permanence_for_active_input() -> None:
    """Active column + active input → permanence increases."""
    cfg = _config(n_columns=10, sparsity=0.2)
    rng = np.random.default_rng(0)
    unit = CorticalUnit(unit_id="u", config=cfg, rng=rng, input_dim=10)

    # Set known state: col 0 connected to input 0
    unit._permanences[:] = 0.0
    unit._permanences[0, 0] = cfg.syn_perm_connected + 0.01  # col 0 connected to input 0
    unit._potential_pool[:] = False
    unit._potential_pool[0, 0] = True

    inp = np.zeros(10, dtype=float)
    inp[0] = 1.0  # input 0 active
    sdr = unit.encode(inp)

    # Force col 0 active in SDR for learning
    sdr.bits[:] = False
    sdr.bits[0] = True

    perm_before = unit._permanences[0, 0]
    unit.learn(sdr)
    assert unit._permanences[0, 0] > perm_before


def test_learn_decreases_permanence_for_inactive_input() -> None:
    """Active column + inactive input → permanence decreases."""
    cfg = _config(n_columns=10, sparsity=0.2)
    rng = np.random.default_rng(0)
    unit = CorticalUnit(unit_id="u", config=cfg, rng=rng, input_dim=10)

    unit._permanences[:] = 0.0
    unit._permanences[0, 1] = 0.6   # col 0 connected to input 1
    unit._potential_pool[:] = False
    unit._potential_pool[0, 1] = True

    inp = np.zeros(10, dtype=float)
    inp[0] = 1.0  # input 1 is INACTIVE
    sdr = unit.encode(inp)

    sdr.bits[:] = False
    sdr.bits[0] = True

    perm_before = unit._permanences[0, 1]
    unit.learn(sdr)
    assert unit._permanences[0, 1] < perm_before


def test_learn_clips_permanence_at_max() -> None:
    """Permanence never exceeds syn_perm_max after learning."""
    cfg = _config(n_columns=10, sparsity=0.2)
    rng = np.random.default_rng(0)
    unit = CorticalUnit(unit_id="u", config=cfg, rng=rng, input_dim=10)

    unit._permanences[:] = 0.0
    unit._permanences[0, 0] = cfg.syn_perm_max  # already at max
    unit._potential_pool[:] = False
    unit._potential_pool[0, 0] = True

    inp = np.ones(10, dtype=float)
    unit.encode(inp)

    sdr = unit.encode(inp)
    sdr.bits[:] = False
    sdr.bits[0] = True
    unit.learn(sdr)

    assert unit._permanences[0, 0] <= cfg.syn_perm_max


def test_learn_trims_below_threshold_to_zero() -> None:
    """Permanences below syn_perm_trim_threshold are set to exactly 0."""
    cfg = _config(n_columns=10, sparsity=0.2)
    rng = np.random.default_rng(0)
    unit = CorticalUnit(unit_id="u", config=cfg, rng=rng, input_dim=10)

    # Set a permanence just above trim threshold
    unit._permanences[:] = 0.0
    unit._permanences[0, 0] = cfg.syn_perm_trim_threshold * 0.5  # below trim
    unit._potential_pool[:] = False
    unit._potential_pool[0, 0] = True

    inp = np.zeros(10, dtype=float)  # input 0 inactive → decrement
    unit.encode(inp)

    sdr = unit.encode(inp)
    sdr.bits[:] = False
    sdr.bits[0] = True
    unit.learn(sdr)

    assert unit._permanences[0, 0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

def test_reset_reinitialises_step_counter() -> None:
    unit = _unit()
    unit.encode(_random_input())
    unit.encode(_random_input(seed=1))
    assert unit._step == 2
    unit.reset()
    assert unit._step == 0


def test_reset_reinitialises_boost_factors() -> None:
    unit = _unit()
    unit._boost_factors[:] = 5.0
    unit.reset()
    np.testing.assert_array_equal(unit._boost_factors, np.ones(unit._config.n_columns))


# ---------------------------------------------------------------------------
# temporal_step() — cell-level output
# ---------------------------------------------------------------------------

def _unit_tm(n_columns: int = 20, cells_per_column: int = 4) -> CorticalUnit:
    """Small unit for TM tests: 20 columns × 4 cells, input_dim=20."""
    cfg = _config(
        n_columns=n_columns,
        sparsity=0.1,        # k=2 winners
        cells_per_column=cells_per_column,
        activation_threshold=2,
        min_threshold=1,
        max_new_synapse_count=4,
        initial_permanence=0.21,
        permanence_increment=0.1,
        permanence_decrement=0.1,
        predicted_segment_decrement=0.0,
    )
    rng = np.random.default_rng(7)
    return CorticalUnit(unit_id="tm", config=cfg, rng=rng, input_dim=20)


def _col_sdr(unit: CorticalUnit, active_cols: list[int]) -> "SDR":
    from halo.core.sdr import SDR
    bits = np.zeros(unit._config.n_columns, dtype=bool)
    for c in active_cols:
        bits[c] = True
    return SDR(bits=bits, unit_id=unit.unit_id, timestamp=0)


def test_tm_total_cell_count() -> None:
    """_segments has exactly n_columns * cells_per_column entries at init."""
    unit = _unit_tm(n_columns=20, cells_per_column=4)
    assert len(unit._segments) == 20 * 4


def test_tm_burst_activates_all_cells_in_column() -> None:
    """An unpredicted active column activates all cells_per_column cells."""
    unit = _unit_tm()
    n_cpc = unit._config.cells_per_column
    col = 3
    sdr = _col_sdr(unit, [col])
    cell_sdr = unit.temporal_step(sdr)
    col_bits = cell_sdr.bits[col * n_cpc : (col + 1) * n_cpc]
    assert col_bits.all(), "All cells in a bursting column must be active"


def test_tm_burst_winner_is_deterministic() -> None:
    """Same config + seed → same winner cell for a bursting column."""
    def run_winner(col: int) -> int:
        unit = _unit_tm()
        sdr = _col_sdr(unit, [col])
        unit.temporal_step(sdr)
        return next(iter(unit._winner_cells))

    w1 = run_winner(5)
    w2 = run_winner(5)
    assert w1 == w2


def test_tm_predicted_column_activates_only_predicted_cells() -> None:
    """A correctly predicted column activates only the predictive cells."""
    unit = _unit_tm()
    n_cpc = unit._config.cells_per_column
    col = 0
    predicted_cell = col * n_cpc + 1  # inject cell 1 of col 0 as predictive

    unit._predictive_cells = {predicted_cell}
    unit._learning_seg_for_predicted = {predicted_cell: None}

    cell_sdr = unit.temporal_step(_col_sdr(unit, [col]))

    col_bits = cell_sdr.bits[col * n_cpc : (col + 1) * n_cpc]
    assert col_bits.sum() == 1, "Only predicted cell should be active"
    assert cell_sdr.bits[predicted_cell], "The predicted cell must be active"


def test_tm_one_winner_per_burst_column() -> None:
    """Each bursting column produces exactly one winner cell."""
    unit = _unit_tm()
    active_cols = [0, 1, 3]
    unit.temporal_step(_col_sdr(unit, active_cols))
    # No predictive cells → all columns burst → one winner each
    assert len(unit._winner_cells) == len(active_cols)


# ---------------------------------------------------------------------------
# learn() — TM AdaptSegments / synapse growth
# ---------------------------------------------------------------------------

def test_tm_learn_reinforces_synapse_to_prev_winner() -> None:
    """Synapse to a prev_winner_cell increases after _adapt_segments()."""
    unit = _unit_tm()
    n_cpc = unit._config.cells_per_column
    cell = 0
    pre = n_cpc  # cell 0 in column 1
    unit._segments[cell] = [{pre: 0.5}]
    unit._prev_winner_cells = {pre}
    unit._winner_cells = {cell}
    unit._learning_seg_for_winner = {cell: 0}

    perm_before = unit._segments[cell][0][pre]
    unit._adapt_segments()
    assert unit._segments[cell][0][pre] > perm_before


def test_tm_learn_grows_synapses_when_no_active_segment() -> None:
    """Winner with no learning segment gets a new segment with synapses."""
    unit = _unit_tm()
    n_cpc = unit._config.cells_per_column
    winner_cell = 0
    prev_winner = n_cpc
    unit._prev_winner_cells = {prev_winner}
    unit._winner_cells = {winner_cell}
    unit._learning_seg_for_winner = {winner_cell: None}

    assert len(unit._segments[winner_cell]) == 0
    unit._adapt_segments()
    assert len(unit._segments[winner_cell]) == 1
    assert prev_winner in unit._segments[winner_cell][0]


def test_tm_no_synapses_grown_on_first_step() -> None:
    """If prev_winner_cells is empty (first step), no segments are created."""
    unit = _unit_tm()
    winner_cell = 0
    unit._prev_winner_cells = set()
    unit._winner_cells = {winner_cell}
    unit._learning_seg_for_winner = {winner_cell: None}

    unit._adapt_segments()
    assert len(unit._segments[winner_cell]) == 0


def test_tm_reset_clears_temporal_state() -> None:
    """After reset(), all TM state is cleared."""
    unit = _unit_tm()
    unit.temporal_step(_col_sdr(unit, [0, 1]))
    assert unit._active_cells  # something was set

    unit.reset()
    assert len(unit._segments) == unit._config.n_columns * unit._config.cells_per_column
    assert all(len(segs) == 0 for segs in unit._segments)
    assert unit._active_cells == set()
    assert unit._winner_cells == set()
    assert unit._predictive_cells == set()
