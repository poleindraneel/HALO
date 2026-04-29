"""Tests for CategoryEncoder."""

from __future__ import annotations

import numpy as np
import pytest

from halo.encoders.category import CategoryEncoder


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

CATS = ["cat", "dog", "bird", "fish"]


def _enc(categories: list[str] | None = None, n: int | None = None, w: int = 4) -> CategoryEncoder:
    cats = categories if categories is not None else CATS
    n_val = n if n is not None else len(cats) * w + 4  # small padding
    return CategoryEncoder(n=n_val, w=w, categories=cats)


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

def test_w_zero_raises() -> None:
    with pytest.raises(ValueError, match="w must be"):
        CategoryEncoder(n=20, w=0, categories=["a", "b"])


def test_empty_categories_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        CategoryEncoder(n=20, w=4, categories=[])


def test_duplicate_categories_raises() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        CategoryEncoder(n=40, w=4, categories=["a", "b", "a"])


def test_n_too_small_raises() -> None:
    # 4 categories × w=4 = 16 bits required; n=15 is too small
    with pytest.raises(ValueError, match="n must be"):
        CategoryEncoder(n=15, w=4, categories=["a", "b", "c", "d"])


def test_n_exact_minimum_valid() -> None:
    enc = CategoryEncoder(n=16, w=4, categories=["a", "b", "c", "d"])
    assert enc.n == 16


# ------------------------------------------------------------------
# Basic properties
# ------------------------------------------------------------------

def test_n_property() -> None:
    enc = _enc()
    assert enc.n == len(CATS) * 4 + 4


def test_w_property() -> None:
    enc = _enc(w=5)
    assert enc.w == 5


def test_categories_property() -> None:
    enc = _enc()
    assert enc.categories == CATS


def test_exactly_w_bits_active() -> None:
    enc = _enc()
    for cat in CATS:
        sdr = enc.encode(cat)
        assert int(sdr.bits.sum()) == 4, f"Failed for category {cat!r}"


def test_sdr_length_equals_n() -> None:
    enc = _enc()
    sdr = enc.encode(CATS[0])
    assert sdr.n == enc.n


def test_unit_id_and_timestamp_propagated() -> None:
    enc = _enc()
    sdr = enc.encode("cat", unit_id="my_unit", timestamp=7)
    assert sdr.unit_id == "my_unit"
    assert sdr.timestamp == 7


# ------------------------------------------------------------------
# Semantic correctness
# ------------------------------------------------------------------

def test_categories_have_zero_overlap() -> None:
    """Each category must map to a non-overlapping pattern."""
    enc = _enc()
    sdrs = {cat: enc.encode(cat) for cat in CATS}
    for i, cat_a in enumerate(CATS):
        for cat_b in CATS[i + 1:]:
            assert sdrs[cat_a].overlap(sdrs[cat_b]) == 0, (
                f"Unexpected overlap between {cat_a!r} and {cat_b!r}"
            )


def test_same_category_identical_sdr() -> None:
    enc = _enc()
    sdr1 = enc.encode("dog")
    sdr2 = enc.encode("dog")
    assert np.array_equal(sdr1.bits, sdr2.bits)


def test_first_category_activates_first_bucket() -> None:
    """First category → bits [0, w)."""
    enc = CategoryEncoder(n=20, w=4, categories=["a", "b", "c", "d"])
    sdr = enc.encode("a")
    assert set(sdr.active_indices.tolist()) == {0, 1, 2, 3}


def test_second_category_activates_second_bucket() -> None:
    """Second category → bits [w, 2w)."""
    enc = CategoryEncoder(n=20, w=4, categories=["a", "b", "c", "d"])
    sdr = enc.encode("b")
    assert set(sdr.active_indices.tolist()) == {4, 5, 6, 7}


# ------------------------------------------------------------------
# Error handling
# ------------------------------------------------------------------

def test_unknown_category_raises_key_error() -> None:
    enc = _enc()
    with pytest.raises(KeyError, match="Unknown category"):
        enc.encode("elephant")
