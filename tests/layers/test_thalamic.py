"""Tests for halo.layers.thalamic.ThalamicLayer (issue #10)."""

from __future__ import annotations

import numpy as np
import pytest

from halo.config.schema import ThalamicConfig
from halo.core.sdr import SDR
from halo.layers.thalamic import ThalamicLayer
from halo.reliability.module import ReliabilityModule
from halo.config.schema import ReliabilityConfig


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _sdr(bits: list[int], n: int = 10, unit_id: str = "u0", ts: int = 1) -> SDR:
    arr = np.zeros(n, dtype=bool)
    for b in bits:
        arr[b] = True
    return SDR(bits=arr, unit_id=unit_id, timestamp=ts)


def _or_layer() -> ThalamicLayer:
    return ThalamicLayer(ThalamicConfig(aggregation="or"))


def _ws_layer(sparsity: float = 0.2) -> ThalamicLayer:
    return ThalamicLayer(ThalamicConfig(aggregation="weighted_sum", output_sparsity=sparsity))


# ---------------------------------------------------------------------------
# ThalamicConfig validation
# ---------------------------------------------------------------------------

def test_config_or_valid() -> None:
    cfg = ThalamicConfig(aggregation="or")
    assert cfg.aggregation == "or"


def test_config_weighted_sum_valid() -> None:
    cfg = ThalamicConfig(aggregation="weighted_sum", output_sparsity=0.05)
    assert cfg.output_sparsity == 0.05


def test_config_invalid_aggregation() -> None:
    with pytest.raises(ValueError, match="aggregation must be one of"):
        ThalamicConfig(aggregation="mean")


def test_config_invalid_sparsity_zero() -> None:
    with pytest.raises(ValueError, match="output_sparsity must be in"):
        ThalamicConfig(aggregation="weighted_sum", output_sparsity=0.0)


def test_config_invalid_sparsity_one() -> None:
    with pytest.raises(ValueError, match="output_sparsity must be in"):
        ThalamicConfig(aggregation="weighted_sum", output_sparsity=1.0)


# ---------------------------------------------------------------------------
# OR mode
# ---------------------------------------------------------------------------

def test_or_single_input() -> None:
    layer = _or_layer()
    sdr = _sdr([2, 5])
    result = layer.aggregate([sdr])
    assert list(np.where(result.bits)[0]) == [2, 5]


def test_or_union_two_sdrs() -> None:
    layer = _or_layer()
    a = _sdr([0, 1], unit_id="u0")
    b = _sdr([1, 2], unit_id="u1")
    result = layer.aggregate([a, b])
    assert set(np.where(result.bits)[0]) == {0, 1, 2}


def test_or_unit_id_is_thalamic() -> None:
    layer = _or_layer()
    result = layer.aggregate([_sdr([0])])
    assert result.unit_id == "thalamic"


def test_or_timestamp_is_max() -> None:
    layer = _or_layer()
    a = SDR(bits=np.array([True, False]), unit_id="u0", timestamp=3)
    b = SDR(bits=np.array([False, True]), unit_id="u1", timestamp=7)
    result = layer.aggregate([a, b])
    assert result.timestamp == 7


def test_or_process_returns_single_element_list() -> None:
    layer = _or_layer()
    result = layer.process([_sdr([0, 3]), _sdr([3, 4], unit_id="u1")])
    assert len(result) == 1
    assert isinstance(result[0], SDR)


def test_or_empty_inputs_warning(caplog) -> None:
    import logging
    layer = _or_layer()
    with caplog.at_level(logging.WARNING, logger="halo.layers.thalamic"):
        out = layer.process([])
    assert out == []
    assert "empty" in caplog.text.lower()


def test_or_aggregate_empty_raises() -> None:
    with pytest.raises(ValueError, match="Cannot aggregate empty list"):
        _or_layer().aggregate([])


# ---------------------------------------------------------------------------
# weighted_sum mode — basic correctness
# ---------------------------------------------------------------------------

def test_weighted_sum_output_sparsity_respected() -> None:
    """Output must have exactly k = round(n * output_sparsity) active bits."""
    n = 20
    sparsity = 0.2  # k = 4
    layer = _ws_layer(sparsity)
    sdrs = [_sdr(list(range(i, i + 5)), n=n, unit_id=f"u{i}") for i in range(3)]
    result = layer.aggregate(sdrs)
    k_expected = round(n * sparsity)
    assert int(result.bits.sum()) == k_expected


def test_weighted_sum_all_zero_weights_returns_empty_sdr(caplog) -> None:
    """When every reliability weight is 0, return an all-inactive SDR.

    Activating k arbitrary bits from an all-zero accumulator would introduce
    noise with no informational basis.  Consistent with ConsensusEngine which
    also emits an empty SDR when total_weight == 0.
    """
    import logging
    n = 10
    layer = _ws_layer(sparsity=0.2)
    a = _sdr([0, 1], n=n, unit_id="a")
    b = _sdr([5, 6], n=n, unit_id="b")
    with caplog.at_level(logging.WARNING, logger="halo.layers.thalamic"):
        result = layer.aggregate([a, b], weights={"a": 0.0, "b": 0.0})
    assert int(result.bits.sum()) == 0, "Expected empty SDR when all weights are zero"
    assert "total_weight=0" in caplog.text


def test_weighted_sum_unit_id_is_thalamic() -> None:
    layer = _ws_layer()
    result = layer.aggregate([_sdr([0, 1], n=10)])
    assert result.unit_id == "thalamic"


def test_weighted_sum_high_weight_unit_dominates() -> None:
    """A unit with weight 10× higher should dominate which bits survive.

    SDR A covers bits {0,1}; SDR B covers bits {5,6}.
    With weight(A)=10, weight(B)=1 and k=2, the output should contain
    bits from A, not B.
    """
    n = 10
    layer = _ws_layer(sparsity=0.2)  # k=2
    a = _sdr([0, 1], n=n, unit_id="a")
    b = _sdr([5, 6], n=n, unit_id="b")
    weights = {"a": 10.0, "b": 1.0}
    result = layer.aggregate([a, b], weights=weights)
    active = set(np.where(result.bits)[0])
    assert active == {0, 1}, f"Expected {{0,1}} dominated by high-weight unit, got {active}"


def test_weighted_sum_equal_weights_symmetric() -> None:
    """With equal weights, aggregation is symmetric in input order."""
    n = 10
    layer = _ws_layer(sparsity=0.2)
    a = _sdr([0, 1], n=n, unit_id="a")
    b = _sdr([0, 1], n=n, unit_id="b")
    r1 = layer.aggregate([a, b])
    r2 = layer.aggregate([b, a])
    np.testing.assert_array_equal(r1.bits, r2.bits)


def test_weighted_sum_zero_weight_unit_suppressed() -> None:
    """A unit with weight 0 contributes nothing; its bits must not appear
    unless another unit also activates them."""
    n = 10
    layer = _ws_layer(sparsity=0.2)  # k=2
    a = _sdr([0, 1], n=n, unit_id="a")
    b = _sdr([5, 6], n=n, unit_id="b")
    weights = {"a": 1.0, "b": 0.0}
    result = layer.aggregate([a, b], weights=weights)
    active = set(np.where(result.bits)[0])
    # b's bits (5,6) should not appear since its weight is 0
    assert not active.intersection({5, 6}), (
        f"Zero-weight unit b should be suppressed, got active={active}"
    )


# ---------------------------------------------------------------------------
# process() with reliability source
# ---------------------------------------------------------------------------

def _reliability_module(scores: dict[str, float]) -> ReliabilityModule:
    cfg = ReliabilityConfig(
        initial_score=0.5,
        alpha=0.1,
        min_score=0.0,
        max_score=1.0,
    )
    mod = ReliabilityModule(list(scores), cfg)
    # Override initial scores by applying signals
    for uid, target in scores.items():
        # Apply enough signal steps to reach target exactly
        mod._scores[uid] = target  # direct set for test setup
    return mod


def test_process_with_reliability_module() -> None:
    """process() with a ReliabilityModule routes weights from get_score()."""
    n = 10
    layer = _ws_layer(sparsity=0.2)  # k=2
    a = _sdr([0, 1], n=n, unit_id="a")
    b = _sdr([5, 6], n=n, unit_id="b")
    rel = _reliability_module({"a": 0.9, "b": 0.1})
    result = layer.process([a, b], reliability=rel)
    assert len(result) == 1
    active = set(np.where(result[0].bits)[0])
    # High-reliability unit a should dominate
    assert active == {0, 1}, f"Expected a's bits to dominate, got {active}"


def test_process_with_dict_reliability() -> None:
    """process() accepts a plain dict as the reliability source."""
    n = 10
    layer = _ws_layer(sparsity=0.2)
    a = _sdr([0, 1], n=n, unit_id="a")
    b = _sdr([5, 6], n=n, unit_id="b")
    result = layer.process([a, b], reliability={"a": 1.0, "b": 0.0})
    active = set(np.where(result[0].bits)[0])
    assert active == {0, 1}


def test_process_no_reliability_uniform_weights() -> None:
    """process() with no reliability falls back to equal weight 1.0."""
    n = 10
    layer = _ws_layer(sparsity=0.4)  # k=4
    # Both units cover the same 4 bits → those 4 should win
    a = _sdr([0, 1, 2, 3], n=n, unit_id="a")
    b = _sdr([0, 1, 2, 3], n=n, unit_id="b")
    result = layer.process([a, b])
    active = set(np.where(result[0].bits)[0])
    assert active == {0, 1, 2, 3}


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------

def test_reset_is_noop_for_stateless_layer() -> None:
    """reset() on a stateless layer must not raise."""
    layer = _or_layer()
    layer.reset()  # should not raise
