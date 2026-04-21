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


# ---------------------------------------------------------------------------
# SDR.copy — issue #1
# ---------------------------------------------------------------------------

def test_copy_bits_equal() -> None:
    """Copied SDR has identical bit values."""
    original = _make_sdr([0, 3, 7], n=20)
    copied = original.copy()
    np.testing.assert_array_equal(original.bits, copied.bits)


def test_copy_independent_array() -> None:
    """Mutating the copy does not affect the original."""
    original = _make_sdr([0, 3, 7], n=20)
    copied = original.copy()
    assert copied.bits is not original.bits
    copied.bits[0] = False
    assert original.bits[0] is np.bool_(True)


def test_copy_preserves_metadata() -> None:
    """unit_id and timestamp are preserved in the copy."""
    original = SDR.from_indices(np.array([1, 2]), n=10, unit_id="u_test", timestamp=42)
    copied = original.copy()
    assert copied.unit_id == "u_test"
    assert copied.timestamp == 42

