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


# ---------------------------------------------------------------------------
# SDR.intersection — issue #2
# ---------------------------------------------------------------------------

def test_intersection_disjoint_is_empty() -> None:
    """Intersection of disjoint SDRs has no active bits."""
    a = _make_sdr([0, 1, 2], n=10)
    b = _make_sdr([7, 8, 9], n=10)
    result = a.intersection(b)
    assert int(result.bits.sum()) == 0


def test_intersection_overlapping() -> None:
    """Intersection contains exactly the shared active bits."""
    a = _make_sdr([0, 1, 2, 3], n=10)
    b = _make_sdr([2, 3, 4, 5], n=10)
    result = a.intersection(b)
    np.testing.assert_array_equal(np.sort(result.active_indices), [2, 3])


def test_intersection_identical() -> None:
    """Intersection of an SDR with itself equals itself."""
    a = _make_sdr([1, 4, 7], n=10)
    result = a.intersection(a)
    np.testing.assert_array_equal(result.bits, a.bits)


def test_intersection_inherits_metadata() -> None:
    """Result carries unit_id and timestamp from self."""
    a = SDR.from_indices(np.array([0]), n=5, unit_id="src", timestamp=7)
    b = SDR.from_indices(np.array([0]), n=5, unit_id="other", timestamp=99)
    result = a.intersection(b)
    assert result.unit_id == "src"
    assert result.timestamp == 7


def test_intersection_length_mismatch_raises() -> None:
    a = SDR.empty(10, "a", 0)
    b = SDR.empty(20, "b", 0)
    with pytest.raises(ValueError):
        a.intersection(b)


# ---------------------------------------------------------------------------
# SDR.normalized_overlap — issue #3
# ---------------------------------------------------------------------------

def test_normalized_overlap_identical() -> None:
    """Normalized overlap of an SDR with itself is 1.0."""
    a = _make_sdr([0, 1, 2, 3], n=10)
    assert a.normalized_overlap(a) == pytest.approx(1.0)


def test_normalized_overlap_disjoint() -> None:
    """Normalized overlap of disjoint SDRs is 0.0."""
    a = _make_sdr([0, 1], n=10)
    b = _make_sdr([5, 6], n=10)
    assert a.normalized_overlap(b) == pytest.approx(0.0)


def test_normalized_overlap_partial() -> None:
    """Partial overlap: 2 shared out of min(3, 4) = 3 → 2/3."""
    a = _make_sdr([0, 1, 2], n=10)
    b = _make_sdr([1, 2, 5, 6], n=10)
    assert a.normalized_overlap(b) == pytest.approx(2 / 3)


def test_normalized_overlap_zero_active_returns_zero() -> None:
    """If either SDR has no active bits, result is 0.0."""
    a = SDR.empty(10, "a", 0)
    b = _make_sdr([1, 2], n=10)
    assert a.normalized_overlap(b) == pytest.approx(0.0)
    assert b.normalized_overlap(a) == pytest.approx(0.0)


def test_normalized_overlap_in_range() -> None:
    """Result is always in [0.0, 1.0]."""
    a = _make_sdr([0, 1, 2, 3, 4], n=20)
    b = _make_sdr([3, 4, 5, 6, 7], n=20)
    score = a.normalized_overlap(b)
    assert 0.0 <= score <= 1.0


def test_normalized_overlap_length_mismatch_raises() -> None:
    a = SDR.empty(10, "a", 0)
    b = SDR.empty(20, "b", 0)
    with pytest.raises(ValueError):
        a.normalized_overlap(b)


# ---------------------------------------------------------------------------
# SDR.subsample — issue #4
# ---------------------------------------------------------------------------

def test_subsample_active_count() -> None:
    """Subsampled SDR has exactly k active bits."""
    rng = np.random.default_rng(0)
    a = _make_sdr([0, 1, 2, 3, 4, 5], n=20)
    result = a.subsample(3, rng)
    assert int(result.bits.sum()) == 3


def test_subsample_bits_subset_of_original() -> None:
    """All active bits in subsample are active in the original."""
    rng = np.random.default_rng(1)
    a = _make_sdr([2, 5, 8, 11, 14], n=20)
    result = a.subsample(2, rng)
    for idx in result.active_indices:
        assert a.bits[idx], f"Index {idx} is not active in original"


def test_subsample_k_equals_active_count() -> None:
    """subsample(k=active_count) returns an equivalent SDR."""
    rng = np.random.default_rng(0)
    a = _make_sdr([1, 3, 5], n=10)
    result = a.subsample(3, rng)
    assert int(result.bits.sum()) == 3
    assert set(result.active_indices.tolist()) == {1, 3, 5}


def test_subsample_k_too_large_raises() -> None:
    rng = np.random.default_rng(0)
    a = _make_sdr([0, 1], n=10)
    with pytest.raises(ValueError):
        a.subsample(5, rng)


def test_subsample_k_zero_raises() -> None:
    rng = np.random.default_rng(0)
    a = _make_sdr([0, 1, 2], n=10)
    with pytest.raises(ValueError):
        a.subsample(0, rng)


def test_subsample_preserves_n() -> None:
    """Subsampled SDR has the same total length n."""
    rng = np.random.default_rng(0)
    a = _make_sdr([0, 5, 10, 15], n=20)
    assert a.subsample(2, rng).n == 20


# ---------------------------------------------------------------------------
# SDR.noise — issue #5
# ---------------------------------------------------------------------------

def test_noise_p_zero_unchanged() -> None:
    """noise(p=0) returns an SDR identical to the original."""
    rng = np.random.default_rng(0)
    a = _make_sdr([1, 3, 5], n=10)
    result = a.noise(0.0, rng)
    np.testing.assert_array_equal(result.bits, a.bits)


def test_noise_p_one_fully_flipped() -> None:
    """noise(p=1) flips every bit."""
    rng = np.random.default_rng(0)
    a = _make_sdr([1, 3, 5], n=10)
    result = a.noise(1.0, rng)
    np.testing.assert_array_equal(result.bits, ~a.bits)


def test_noise_does_not_mutate_original() -> None:
    """noise() returns a new SDR; original bits are unchanged."""
    rng = np.random.default_rng(0)
    a = _make_sdr([0, 1, 2], n=10)
    original_bits = a.bits.copy()
    a.noise(0.5, rng)
    np.testing.assert_array_equal(a.bits, original_bits)


def test_noise_preserves_n() -> None:
    rng = np.random.default_rng(0)
    a = _make_sdr([0, 1], n=20)
    assert a.noise(0.3, rng).n == 20


def test_noise_invalid_p_raises() -> None:
    rng = np.random.default_rng(0)
    a = _make_sdr([0], n=5)
    with pytest.raises(ValueError):
        a.noise(1.5, rng)
    with pytest.raises(ValueError):
        a.noise(-0.1, rng)


# ---------------------------------------------------------------------------
# SDR.hamming_distance — issue #6
# ---------------------------------------------------------------------------

def test_hamming_distance_identical_is_zero() -> None:
    """Hamming distance of an SDR with itself is 0."""
    a = _make_sdr([0, 3, 7], n=10)
    assert a.hamming_distance(a) == 0


def test_hamming_distance_disjoint() -> None:
    """Hamming distance of disjoint SDRs equals sum of both active counts."""
    a = _make_sdr([0, 1, 2], n=10)
    b = _make_sdr([7, 8, 9], n=10)
    assert a.hamming_distance(b) == 6


def test_hamming_distance_partial_overlap() -> None:
    """Partial overlap: bits that differ = non-shared bits in each."""
    a = _make_sdr([0, 1, 2], n=10)    # 3 active
    b = _make_sdr([1, 2, 3], n=10)    # 3 active; 2 shared, 1 unique each
    assert a.hamming_distance(b) == 2  # bit 0 and bit 3 differ


def test_hamming_distance_symmetric() -> None:
    """hamming_distance is symmetric."""
    a = _make_sdr([0, 1, 5], n=10)
    b = _make_sdr([1, 4, 5], n=10)
    assert a.hamming_distance(b) == b.hamming_distance(a)


def test_hamming_distance_length_mismatch_raises() -> None:
    a = SDR.empty(10, "a", 0)
    b = SDR.empty(20, "b", 0)
    with pytest.raises(ValueError):
        a.hamming_distance(b)


# ---------------------------------------------------------------------------
# metrics.overlap_score delegates to normalized_overlap — issue #3
# ---------------------------------------------------------------------------

def test_metrics_overlap_score_delegates() -> None:
    """metrics.overlap_score must return the same value as normalized_overlap."""
    from halo.utils.metrics import overlap_score
    a = _make_sdr([0, 1, 2], n=10)
    b = _make_sdr([1, 2, 3], n=10)
    assert overlap_score(a, b) == pytest.approx(a.normalized_overlap(b))

