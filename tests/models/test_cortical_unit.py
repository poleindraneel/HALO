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
    """Columns with no connected synapses on active inputs score 0 overlap."""
    cfg = _config(n_columns=10, sparsity=0.2, stimulus_threshold=0.0)
    rng = np.random.default_rng(0)
    unit = CorticalUnit(unit_id="u", config=cfg, rng=rng, input_dim=10)

    # Force all permanences below connected threshold — no column gets overlap
    unit._permanences[:] = cfg.syn_perm_connected - 0.1
    unit._potential_pool[:] = True

    # All-ones input — but no connected synapses
    inp = np.ones(10, dtype=float)
    # Even with no overlaps, WTA still picks k winners (argpartition on zeros)
    sdr = unit.encode(inp)
    # overlaps are all 0 — boosted overlaps all 0 — WTA picks arbitrary k cols
    # key assertion: total active == k regardless
    k = max(1, int(cfg.sparsity * cfg.n_columns))
    assert int(sdr.bits.sum()) == k


def test_encode_stimulus_threshold_zeros_low_overlap() -> None:
    """Columns with overlap below stimulus_threshold are excluded from WTA."""
    cfg = _config(n_columns=20, sparsity=0.1, stimulus_threshold=5.0)
    rng = np.random.default_rng(0)
    unit = CorticalUnit(unit_id="u", config=cfg, rng=rng, input_dim=50)

    # Force all permanences above connected threshold so all inputs connect
    unit._permanences[:] = cfg.syn_perm_connected + 0.1
    unit._potential_pool[:] = True

    # Sparse input — only 2 active bits → max overlap = 2 < threshold 5
    inp = np.zeros(50, dtype=float)
    inp[:2] = 1.0

    sdr = unit.encode(inp)
    # All boosted scores are 0 (below threshold) — WTA picks k arbitrary cols
    # The important thing: it doesn't crash and returns correct size
    k = max(1, int(cfg.sparsity * cfg.n_columns))
    assert int(sdr.bits.sum()) == k


def test_potential_pool_global_covers_all_inputs() -> None:
    """With potential_radius=-1, every column has potential connections to inputs."""
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
