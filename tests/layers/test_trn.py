"""Tests for halo.layers.trn.TRNGatingLayer."""

import numpy as np
import pytest

from halo.config.schema import TRNConfig
from halo.core.sdr import SDR
from halo.layers.trn import TRNGatingLayer


def _constant_sdr(value: bool, n: int = 100, uid: str = "u", ts: int = 0) -> SDR:
    bits = np.full(n, value, dtype=bool)
    return SDR(bits=bits, unit_id=uid, timestamp=ts)


def _sdr_from_seed(seed: int, n: int = 100, sparsity: float = 0.1, ts: int = 0) -> SDR:
    rng = np.random.default_rng(seed)
    bits = rng.random(n) < sparsity
    return SDR(bits=bits, unit_id=f"u_{seed}", timestamp=ts)


def test_no_gating_low_entropy() -> None:
    """Identical SDRs produce zero entropy → all SDRs pass through unchanged."""
    cfg = TRNConfig(entropy_threshold=0.5)
    layer = TRNGatingLayer(cfg)

    # All identical SDRs → entropy = 0
    sdr = _constant_sdr(False, n=100, uid="u0")
    sdrs = [
        SDR(bits=sdr.bits.copy(), unit_id=f"u{i}", timestamp=0) for i in range(4)
    ]
    # Make them non-trivially identical (some active bits)
    for s in sdrs:
        s.bits[:10] = True

    result = layer.process(sdrs)
    assert len(result) == len(sdrs)
    for original, gated in zip(sdrs, result):
        np.testing.assert_array_equal(original.bits, gated.bits)


def test_gating_high_entropy() -> None:
    """Highly varied SDRs → high entropy → at least one SDR suppressed."""
    cfg = TRNConfig(entropy_threshold=0.01)  # very low threshold to ensure gating
    layer = TRNGatingLayer(cfg)

    # Create SDRs with varying sparsity so some are below mean
    sdrs = []
    rng = np.random.default_rng(0)
    for i in range(8):
        bits = rng.random(200) < (0.05 * (i + 1))
        sdrs.append(SDR(bits=bits, unit_id=f"u{i}", timestamp=0))

    result = layer.process(sdrs)
    assert len(result) == len(sdrs)

    # At least one SDR should be suppressed (all-zero)
    suppressed_count = sum(1 for s in result if int(s.bits.sum()) == 0)
    assert suppressed_count >= 1


def test_empty_input_returns_empty() -> None:
    cfg = TRNConfig(entropy_threshold=0.5)
    layer = TRNGatingLayer(cfg)
    assert layer.process([]) == []


def test_single_sdr_passes_through() -> None:
    """A single SDR cannot have high cross-population entropy; it always passes."""
    cfg = TRNConfig(entropy_threshold=0.5)
    layer = TRNGatingLayer(cfg)
    bits = np.zeros(50, dtype=bool)
    bits[:5] = True
    sdr = SDR(bits=bits, unit_id="solo", timestamp=0)
    result = layer.process([sdr])
    assert len(result) == 1
    np.testing.assert_array_equal(result[0].bits, sdr.bits)
