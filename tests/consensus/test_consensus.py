"""Tests for halo.consensus.engine.ConsensusEngine."""

import numpy as np
import pytest

from halo.config.schema import ConsensusConfig
from halo.consensus.engine import ConsensusEngine
from halo.core.sdr import SDR


def _make_engine() -> ConsensusEngine:
    return ConsensusEngine(ConsensusConfig(method="weighted_vote"))


def _sdr(active: list[int], n: int = 20, uid: str = "u") -> SDR:
    return SDR.from_indices(np.array(active), n=n, unit_id=uid, timestamp=0)


def test_unanimous_vote() -> None:
    """When all units produce the same SDR with equal weights, output == input."""
    engine = _make_engine()
    active = [0, 1, 2, 5, 10]
    sdrs = [_sdr(active, uid=f"unit_{i}") for i in range(4)]
    weights = {f"unit_{i}": 1.0 for i in range(4)}

    result = engine.aggregate(sdrs, weights)
    np.testing.assert_array_equal(result.bits, sdrs[0].bits)


def test_weighted_dominant_unit() -> None:
    """One unit with weight 1.0, others with 0.0 → output == that unit's SDR."""
    engine = _make_engine()
    dominant = _sdr([3, 7, 11], n=20, uid="dominant")
    others = [_sdr([0, 1, 2], n=20, uid=f"other_{i}") for i in range(3)]

    sdrs = [dominant] + others
    weights = {"dominant": 1.0, "other_0": 0.0, "other_1": 0.0, "other_2": 0.0}

    result = engine.aggregate(sdrs, weights)
    np.testing.assert_array_equal(result.bits, dominant.bits)


def test_consensus_unit_id() -> None:
    engine = _make_engine()
    sdrs = [_sdr([0, 1], uid=f"u{i}") for i in range(2)]
    weights = {f"u{i}": 1.0 for i in range(2)}
    result = engine.aggregate(sdrs, weights)
    assert result.unit_id == "consensus"


def test_zero_weights_produce_empty_sdr() -> None:
    """All-zero weights → no bits exceed 0.5 → all-zero consensus."""
    engine = _make_engine()
    sdrs = [_sdr([0, 1, 2], uid=f"u{i}") for i in range(3)]
    weights = {f"u{i}": 0.0 for i in range(3)}
    result = engine.aggregate(sdrs, weights)
    assert int(result.bits.sum()) == 0


def test_empty_sdrs_raises() -> None:
    engine = _make_engine()
    with pytest.raises(ValueError):
        engine.aggregate([], {})
