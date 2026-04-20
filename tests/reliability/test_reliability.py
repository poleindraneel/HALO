"""Tests for halo.reliability.module.ReliabilityModule."""

import pytest

from halo.config.schema import ReliabilityConfig
from halo.reliability.module import ReliabilityModule


def _make_module(initial: float = 0.5) -> ReliabilityModule:
    cfg = ReliabilityConfig(
        initial_score=initial,
        alpha=0.1,
        min_score=0.01,
        max_score=1.0,
    )
    return ReliabilityModule(unit_ids=["unit_0", "unit_1"], config=cfg)


def test_positive_signal_increases_score() -> None:
    mod = _make_module()
    before = mod.get_score("unit_0")
    mod.update("unit_0", signal=1.0)
    assert mod.get_score("unit_0") > before


def test_negative_signal_decreases_score() -> None:
    mod = _make_module()
    before = mod.get_score("unit_0")
    mod.update("unit_0", signal=-1.0)
    assert mod.get_score("unit_0") < before


def test_score_clamped_to_max() -> None:
    mod = _make_module(initial=0.99)
    for _ in range(20):
        mod.update("unit_0", signal=1.0)
    assert mod.get_score("unit_0") <= 1.0


def test_score_clamped_to_min() -> None:
    mod = _make_module(initial=0.02)
    for _ in range(20):
        mod.update("unit_0", signal=-1.0)
    assert mod.get_score("unit_0") >= 0.01


def test_all_scores_returns_all_units() -> None:
    mod = _make_module()
    scores = mod.all_scores()
    assert set(scores.keys()) == {"unit_0", "unit_1"}


def test_all_scores_is_copy() -> None:
    """Mutating the returned dict must not affect internal state."""
    mod = _make_module()
    scores = mod.all_scores()
    scores["unit_0"] = 0.0
    assert mod.get_score("unit_0") == pytest.approx(0.5)


def test_unknown_unit_raises() -> None:
    mod = _make_module()
    with pytest.raises(KeyError):
        mod.get_score("nonexistent")
