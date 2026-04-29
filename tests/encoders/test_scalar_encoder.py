"""Tests for ScalarEncoder."""

from __future__ import annotations

import numpy as np
import pytest

from halo.encoders.scalar import ScalarEncoder


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _enc(n: int = 100, w: int = 10, min_val: float = 0.0, max_val: float = 1.0,
         periodic: bool = False) -> ScalarEncoder:
    return ScalarEncoder(n=n, w=w, min_val=min_val, max_val=max_val, periodic=periodic)


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

def test_w_zero_raises() -> None:
    with pytest.raises(ValueError, match="w must be"):
        ScalarEncoder(n=10, w=0, min_val=0.0, max_val=1.0)


def test_n_equal_w_raises() -> None:
    with pytest.raises(ValueError, match="n must be > w"):
        ScalarEncoder(n=10, w=10, min_val=0.0, max_val=1.0)


def test_n_less_than_w_raises() -> None:
    with pytest.raises(ValueError, match="n must be > w"):
        ScalarEncoder(n=5, w=10, min_val=0.0, max_val=1.0)


def test_min_ge_max_raises() -> None:
    with pytest.raises(ValueError, match="min_val must be < max_val"):
        ScalarEncoder(n=100, w=10, min_val=1.0, max_val=0.0)


def test_min_equal_max_raises() -> None:
    with pytest.raises(ValueError, match="min_val must be < max_val"):
        ScalarEncoder(n=100, w=10, min_val=0.5, max_val=0.5)


# ------------------------------------------------------------------
# Basic properties
# ------------------------------------------------------------------

def test_n_property() -> None:
    enc = _enc(n=100, w=10)
    assert enc.n == 100


def test_w_property() -> None:
    enc = _enc(n=100, w=10)
    assert enc.w == 10


def test_exactly_w_bits_active() -> None:
    enc = _enc(n=100, w=10)
    sdr = enc.encode(0.5)
    assert int(sdr.bits.sum()) == 10


def test_sdr_length_equals_n() -> None:
    enc = _enc(n=100, w=10)
    sdr = enc.encode(0.5)
    assert sdr.n == 100


def test_unit_id_and_timestamp_propagated() -> None:
    enc = _enc()
    sdr = enc.encode(0.5, unit_id="u0", timestamp=42)
    assert sdr.unit_id == "u0"
    assert sdr.timestamp == 42


# ------------------------------------------------------------------
# Non-periodic: boundary and clamping
# ------------------------------------------------------------------

def test_min_value_activates_first_w_bits() -> None:
    """min_val should produce a window starting at index 0."""
    enc = _enc(n=100, w=10, min_val=0.0, max_val=1.0)
    sdr = enc.encode(0.0)
    assert set(sdr.active_indices.tolist()) == set(range(10))


def test_max_value_activates_last_w_bits() -> None:
    """max_val should produce a window ending at index n-1."""
    enc = _enc(n=100, w=10, min_val=0.0, max_val=1.0)
    sdr = enc.encode(1.0)
    assert set(sdr.active_indices.tolist()) == set(range(90, 100))


def test_below_min_clamped_to_min() -> None:
    enc = _enc(n=100, w=10, min_val=0.0, max_val=1.0)
    sdr_min = enc.encode(0.0)
    sdr_below = enc.encode(-99.0)
    assert np.array_equal(sdr_min.bits, sdr_below.bits)


def test_above_max_clamped_to_max() -> None:
    enc = _enc(n=100, w=10, min_val=0.0, max_val=1.0)
    sdr_max = enc.encode(1.0)
    sdr_above = enc.encode(99.0)
    assert np.array_equal(sdr_max.bits, sdr_above.bits)


# ------------------------------------------------------------------
# Non-periodic: semantic overlap
# ------------------------------------------------------------------

def test_nearby_values_high_overlap() -> None:
    """Values close together should share most active bits."""
    enc = _enc(n=200, w=20)
    sdr_a = enc.encode(0.5)
    sdr_b = enc.encode(0.51)
    assert sdr_a.overlap(sdr_b) >= 15


def test_far_values_low_overlap() -> None:
    """Values far apart should share few or no active bits."""
    enc = _enc(n=200, w=20)
    sdr_a = enc.encode(0.0)
    sdr_b = enc.encode(1.0)
    assert sdr_a.overlap(sdr_b) == 0


def test_same_value_identical_sdr() -> None:
    enc = _enc()
    sdr1 = enc.encode(0.3)
    sdr2 = enc.encode(0.3)
    assert np.array_equal(sdr1.bits, sdr2.bits)


# ------------------------------------------------------------------
# Periodic
# ------------------------------------------------------------------

def test_periodic_exactly_w_bits() -> None:
    enc = _enc(n=100, w=10, periodic=True)
    sdr = enc.encode(0.5)
    assert int(sdr.bits.sum()) == 10


def test_periodic_wrap_around_overlap() -> None:
    """Value near max_val should overlap with value near min_val for periodic encoder."""
    enc = ScalarEncoder(n=100, w=20, min_val=0.0, max_val=360.0, periodic=True)
    sdr_near_max = enc.encode(359.0)
    sdr_near_min = enc.encode(1.0)
    # They should overlap (the window wraps from ~index 99 back to 0)
    assert sdr_near_max.overlap(sdr_near_min) > 0


def test_periodic_mid_value_no_wrap() -> None:
    """Mid-range periodic value should not wrap — indices stay within [0, n)."""
    enc = _enc(n=100, w=10, periodic=True)
    sdr = enc.encode(0.5)
    assert sdr.bits.sum() == 10
    assert (sdr.active_indices >= 0).all()
    assert (sdr.active_indices < 100).all()
