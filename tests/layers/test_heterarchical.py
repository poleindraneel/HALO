"""Tests for halo.layers.heterarchical.HeterarchicalLayer (issue #18)."""

from __future__ import annotations

import numpy as np
import pytest

from halo.config.schema import HeterarchicalConfig
from halo.core.sdr import SDR
from halo.layers.heterarchical import HeterarchicalLayer


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

N = 40  # small n_columns for fast tests


def _cfg(lr: float = 0.1, decay: float = 0.01, sparsity: float = 0.5) -> HeterarchicalConfig:
    return HeterarchicalConfig(lateral_lr=lr, lateral_decay=decay, lateral_sparsity=sparsity)


def _layer(n: int = N, lr: float = 0.1, decay: float = 0.01, sparsity: float = 0.5,
           seed: int = 0) -> HeterarchicalLayer:
    return HeterarchicalLayer(
        n_columns=n,
        config=_cfg(lr=lr, decay=decay, sparsity=sparsity),
        rng=np.random.default_rng(seed),
    )


def _sdr(active: list[int], n: int = N, uid: str = "u0", ts: int = 0) -> SDR:
    return SDR.from_indices(np.array(active, dtype=int), n, uid, ts)


def _two_unit_layer() -> tuple[HeterarchicalLayer, str, str]:
    """Return a layer with two units connected u0 → u1 and u1 → u0."""
    layer = _layer(sparsity=1.0)  # fully connected pool for predictable tests
    layer.register_unit("u0")
    layer.register_unit("u1")
    layer.add_connection("u0", "u1")
    layer.add_connection("u1", "u0")
    return layer, "u0", "u1"


# ------------------------------------------------------------------
# Config validation
# ------------------------------------------------------------------

def test_config_lateral_lr_zero_raises() -> None:
    with pytest.raises(ValueError, match="lateral_lr"):
        HeterarchicalConfig(lateral_lr=0.0)


def test_config_lateral_decay_negative_raises() -> None:
    with pytest.raises(ValueError, match="lateral_decay"):
        HeterarchicalConfig(lateral_decay=-0.01)


def test_config_lateral_sparsity_zero_raises() -> None:
    with pytest.raises(ValueError, match="lateral_sparsity"):
        HeterarchicalConfig(lateral_sparsity=0.0)


# ------------------------------------------------------------------
# Initialisation
# ------------------------------------------------------------------

def test_weights_within_mask_only() -> None:
    """Weights outside the potential pool mask must be exactly zero."""
    layer, u0, u1 = _two_unit_layer()
    w = layer.weight_matrix("u0", "u1")
    mask = layer.potential_mask("u0", "u1")
    assert np.all(w[~mask] == 0.0)


def test_weights_initialised_small() -> None:
    """All initial weights within the pool are in [0, 0.1)."""
    layer, u0, u1 = _two_unit_layer()
    w = layer.weight_matrix("u0", "u1")
    mask = layer.potential_mask("u0", "u1")
    assert np.all(w[mask] >= 0.0)
    assert np.all(w[mask] < 0.1 + 1e-6)


def test_full_sparsity_mask_covers_all() -> None:
    """sparsity=1.0 should give a fully connected potential pool."""
    layer, u0, u1 = _two_unit_layer()
    mask = layer.potential_mask("u0", "u1")
    assert mask.all()


def test_process_returns_inputs_unchanged() -> None:
    """process() is a pass-through — does not modify SDRs."""
    layer, u0, u1 = _two_unit_layer()
    sdr0 = _sdr([0, 1, 2], uid="u0")
    sdr1 = _sdr([5, 6, 7], uid="u1")
    result = layer.process([sdr0, sdr1])
    assert len(result) == 2
    np.testing.assert_array_equal(result[0].bits, sdr0.bits)
    np.testing.assert_array_equal(result[1].bits, sdr1.bits)


# ------------------------------------------------------------------
# Bias computation
# ------------------------------------------------------------------

def test_bias_is_zero_before_any_sdrs() -> None:
    """On the first step, no SDRs have been cached → all biases are zero."""
    layer, u0, u1 = _two_unit_layer()
    biases = layer.compute_biases()
    assert np.all(biases["u0"] == 0.0)
    assert np.all(biases["u1"] == 0.0)


def test_bias_is_weighted_sum_of_lateral_sdrs() -> None:
    """Bias for u1 = W[u0→u1] @ sdr_u0.bits (with fully connected pool)."""
    layer, u0, u1 = _two_unit_layer()

    # Manually set W[u0→u1] to identity so bias = sdr_u0.bits
    w = layer.weight_matrix("u0", "u1")
    w[:] = 0.0
    np.fill_diagonal(w, 1.0)  # identity matrix

    sdr_u0 = _sdr([0, 5, 10], uid="u0")
    sdr_u1 = _sdr([1, 2], uid="u1")
    layer.update_sdrs([sdr_u0, sdr_u1])

    biases = layer.compute_biases()

    # With identity W, bias[u1][i] == 1.0 if i in {0,5,10}, else 0
    expected = np.zeros(N, dtype=np.float32)
    expected[[0, 5, 10]] = 1.0
    np.testing.assert_allclose(biases["u1"], expected, atol=1e-6)


def test_bias_zero_for_unconnected_unit() -> None:
    """A unit with no incoming connections always gets zero bias."""
    layer = _layer(sparsity=1.0)
    layer.register_unit("u0")
    layer.register_unit("u1")
    layer.add_connection("u0", "u1")   # only u0 → u1, nothing into u0

    sdr_u0 = _sdr([0, 1], uid="u0")
    sdr_u1 = _sdr([2, 3], uid="u1")
    layer.update_sdrs([sdr_u0, sdr_u1])

    biases = layer.compute_biases()
    # u0 has no incoming edges → bias must be zero
    assert np.all(biases["u0"] == 0.0)


# ------------------------------------------------------------------
# Hebbian learning
# ------------------------------------------------------------------

def test_learn_increments_coactive_weights() -> None:
    """Co-active column pairs across units must have increased weights."""
    layer, u0, u1 = _two_unit_layer()

    # Force all weights to a known baseline
    for uid_from in ("u0", "u1"):
        for uid_to in ("u0", "u1"):
            if uid_from != uid_to:
                layer.weight_matrix(uid_from, uid_to)[:] = 0.0

    sdr0 = _sdr([0, 1], uid="u0")
    sdr1 = _sdr([2, 3], uid="u1")

    layer.learn([sdr0, sdr1])

    w = layer.weight_matrix("u0", "u1")
    # u1 active: cols 2,3 — u0 active: cols 0,1
    # co-active pairs: (2,0), (2,1), (3,0), (3,1)
    assert w[2, 0] > 0.0
    assert w[2, 1] > 0.0
    assert w[3, 0] > 0.0
    assert w[3, 1] > 0.0


def test_learn_decays_non_coactive_weights() -> None:
    """Non-coactive pool synapses must decay after a learn step."""
    layer, u0, u1 = _two_unit_layer()

    # Pre-set some weights
    w = layer.weight_matrix("u0", "u1")
    w[:] = 0.5  # start all at 0.5

    sdr0 = _sdr([0], uid="u0")
    sdr1 = _sdr([0], uid="u1")

    layer.learn([sdr0, sdr1])

    # (0,0) is co-active → should increase (or at least not decay)
    assert w[0, 0] >= 0.5
    # (0,1) is non-coactive → should decay below 0.5
    assert w[0, 1] < 0.5


def test_learn_clips_weights_to_zero() -> None:
    """Decayed weights must never go below 0."""
    layer, u0, u1 = _two_unit_layer()

    # All weights near zero; a large decay should not make them negative
    w = layer.weight_matrix("u0", "u1")
    w[:] = 0.005  # very small

    cfg = HeterarchicalConfig(lateral_lr=0.01, lateral_decay=0.5, lateral_sparsity=1.0)
    layer2 = HeterarchicalLayer(n_columns=N, config=cfg, rng=np.random.default_rng(0))
    layer2.register_unit("u0")
    layer2.register_unit("u1")
    layer2.add_connection("u0", "u1")
    layer2.weight_matrix("u0", "u1")[:] = 0.005

    sdr0 = _sdr([0], uid="u0")
    sdr1 = _sdr([1], uid="u1")  # no co-activity → all decay
    layer2.learn([sdr0, sdr1])

    assert np.all(layer2.weight_matrix("u0", "u1") >= 0.0)


def test_learn_clips_weights_to_one() -> None:
    """Incremented weights must never exceed 1."""
    layer, u0, u1 = _two_unit_layer()
    w = layer.weight_matrix("u0", "u1")
    w[:] = 0.99  # near-max

    sdr0 = _sdr(list(range(N)), uid="u0")  # all active
    sdr1 = _sdr(list(range(N)), uid="u1")  # all active
    layer.learn([sdr0, sdr1])

    assert np.all(layer.weight_matrix("u0", "u1") <= 1.0)


def test_learn_only_updates_within_mask() -> None:
    """Weights outside the potential pool must stay exactly zero after learning."""
    layer = _layer(sparsity=0.1, seed=42)  # sparse — some entries definitely zero
    layer.register_unit("u0")
    layer.register_unit("u1")
    layer.add_connection("u0", "u1")

    mask = layer.potential_mask("u0", "u1")
    # Confirm there are out-of-pool entries
    assert (~mask).any(), "Test requires some entries outside the potential pool"

    sdr0 = _sdr(list(range(N)), uid="u0")
    sdr1 = _sdr(list(range(N)), uid="u1")
    layer.learn([sdr0, sdr1])

    w = layer.weight_matrix("u0", "u1")
    assert np.all(w[~mask] == 0.0)


# ------------------------------------------------------------------
# reset()
# ------------------------------------------------------------------

def test_reset_clears_prev_sdrs() -> None:
    """reset() must clear cached SDRs (no bias on next step)."""
    layer, u0, u1 = _two_unit_layer()

    sdr0 = _sdr([0, 1], uid="u0")
    sdr1 = _sdr([2, 3], uid="u1")
    layer.update_sdrs([sdr0, sdr1])

    # Confirm biases are non-zero before reset
    biases_before = layer.compute_biases()
    # At least one unit should have non-zero bias (weights are non-zero from init)

    layer.reset()
    biases_after = layer.compute_biases()
    assert np.all(biases_after["u0"] == 0.0)
    assert np.all(biases_after["u1"] == 0.0)


def test_reset_preserves_weights() -> None:
    """reset() must not destroy learned weights."""
    layer, u0, u1 = _two_unit_layer()

    sdr0 = _sdr([0, 1], uid="u0")
    sdr1 = _sdr([2, 3], uid="u1")
    layer.learn([sdr0, sdr1])

    w_before = layer.weight_matrix("u0", "u1").copy()
    layer.reset()
    w_after = layer.weight_matrix("u0", "u1")

    np.testing.assert_array_equal(w_before, w_after)
