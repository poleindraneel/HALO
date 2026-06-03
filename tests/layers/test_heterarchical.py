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
    identity = np.eye(N, dtype=np.float32)
    layer.set_dense_weights("u0", "u1", identity)

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
                layer.set_all_weights(uid_from, uid_to, 0.0)

    sdr0 = _sdr([0, 1], uid="u0")
    sdr1 = _sdr([2, 3], uid="u1")

    layer.learn([sdr0, sdr1])

    # u1 active: cols 2,3 — u0 active: cols 0,1
    # co-active pairs: (2,0), (2,1), (3,0), (3,1)
    assert layer.get_weight("u0", "u1", 2, 0) > 0.0
    assert layer.get_weight("u0", "u1", 2, 1) > 0.0
    assert layer.get_weight("u0", "u1", 3, 0) > 0.0
    assert layer.get_weight("u0", "u1", 3, 1) > 0.0


def test_learn_decays_non_coactive_weights() -> None:
    """Non-coactive pool synapses must decay after a learn step."""
    layer, u0, u1 = _two_unit_layer()

    # Pre-set all in-pool synapses to 0.5
    layer.set_all_weights("u0", "u1", 0.5)

    sdr0 = _sdr([0], uid="u0")
    sdr1 = _sdr([0], uid="u1")

    layer.learn([sdr0, sdr1])

    # (0,0) is co-active → should increase (or at least not decay)
    assert layer.get_weight("u0", "u1", 0, 0) >= 0.5
    # (0,1) is non-coactive → should decay below 0.5
    assert layer.get_weight("u0", "u1", 0, 1) < 0.5


def test_learn_clips_weights_to_zero() -> None:
    """Decayed weights must never go below 0."""
    cfg = HeterarchicalConfig(lateral_lr=0.01, lateral_decay=0.5, lateral_sparsity=1.0)
    layer2 = HeterarchicalLayer(n_columns=N, config=cfg, rng=np.random.default_rng(0))
    layer2.register_unit("u0")
    layer2.register_unit("u1")
    layer2.add_connection("u0", "u1")
    layer2.set_all_weights("u0", "u1", 0.005)  # pre-set small initial weights

    sdr0 = _sdr([0], uid="u0")
    sdr1 = _sdr([1], uid="u1")  # no co-activity → all decay
    layer2.learn([sdr0, sdr1])

    assert np.all(layer2.weight_matrix("u0", "u1") >= 0.0)


def test_learn_clips_weights_to_one() -> None:
    """Incremented weights must never exceed 1."""
    layer, u0, u1 = _two_unit_layer()
    layer.set_all_weights("u0", "u1", 0.99)  # near-max

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

    # Confirm biases are non-zero before reset (weights are non-zero after init + SDRs cached)
    biases_before = layer.compute_biases()
    assert any(np.any(b != 0.0) for b in biases_before.values()), (
        "Expected non-zero biases after caching SDRs with non-zero weights"
    )

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


# ------------------------------------------------------------------
# CSR-sparse storage (issue #25)
# ------------------------------------------------------------------

def test_csr_arrays_have_consistent_shapes() -> None:
    """indptr length n+1, indices/data/row_idx all length nnz."""
    layer, _, _ = _two_unit_layer()
    indptr, indices, data, row_idx = layer.edge_arrays("u0", "u1")
    assert indptr.shape == (N + 1,)
    assert indices.shape == data.shape == row_idx.shape
    assert int(indptr[-1]) == data.size


def test_csr_row_idx_is_repeat_of_indptr_diff() -> None:
    """row_idx must equal np.repeat(arange(n), diff(indptr))."""
    layer, _, _ = _two_unit_layer()
    indptr, _, _, row_idx = layer.edge_arrays("u0", "u1")
    expected = np.repeat(np.arange(N, dtype=np.int32), np.diff(indptr))
    np.testing.assert_array_equal(row_idx, expected)


def test_csr_indices_sorted_within_each_row() -> None:
    """indices[indptr[r]:indptr[r+1]] must be sorted ascending — required by
    get_weight's binary search to be correct."""
    layer = _layer(sparsity=0.3, seed=99)
    layer.register_unit("u0")
    layer.register_unit("u1")
    layer.add_connection("u0", "u1")
    indptr, indices, _, _ = layer.edge_arrays("u0", "u1")
    for r in range(N):
        row = indices[indptr[r] : indptr[r + 1]]
        assert np.all(np.diff(row) >= 0), f"row {r} not sorted: {row}"


def test_n_synapses_matches_data_size() -> None:
    layer, _, _ = _two_unit_layer()
    assert layer.n_synapses("u0", "u1") == layer.edge_arrays("u0", "u1")[2].size


def test_get_weight_returns_zero_for_out_of_pool_pair() -> None:
    """For sparsity < 1 there must be at least one (to, from) pair outside the pool."""
    layer = _layer(sparsity=0.1, seed=7)
    layer.register_unit("u0")
    layer.register_unit("u1")
    layer.add_connection("u0", "u1")
    mask = layer.potential_mask("u0", "u1")
    out_to, out_from = np.where(~mask)
    assert out_to.size > 0
    assert layer.get_weight("u0", "u1", int(out_to[0]), int(out_from[0])) == 0.0


def test_get_weight_returns_stored_value_for_in_pool_pair() -> None:
    layer, _, _ = _two_unit_layer()  # sparsity=1.0 → all pairs in pool
    indptr, indices, data, row_idx = layer.edge_arrays("u0", "u1")
    # Pick the first synapse and verify get_weight returns its data value.
    to_col, from_col = int(row_idx[0]), int(indices[0])
    assert layer.get_weight("u0", "u1", to_col, from_col) == pytest.approx(float(data[0]))


def test_set_all_weights_uniform_value() -> None:
    layer, _, _ = _two_unit_layer()
    layer.set_all_weights("u0", "u1", 0.42)
    _, _, data, _ = layer.edge_arrays("u0", "u1")
    np.testing.assert_allclose(data, 0.42)


def test_set_all_weights_rejects_out_of_range() -> None:
    layer, _, _ = _two_unit_layer()
    with pytest.raises(ValueError):
        layer.set_all_weights("u0", "u1", 1.5)
    with pytest.raises(ValueError):
        layer.set_all_weights("u0", "u1", -0.1)


def test_set_dense_weights_respects_pool_mask() -> None:
    """set_dense_weights must ignore values outside the potential pool."""
    layer = _layer(sparsity=0.3, seed=11)
    layer.register_unit("u0")
    layer.register_unit("u1")
    layer.add_connection("u0", "u1")

    # Fill a dense array with 0.7 everywhere
    dense = np.full((N, N), 0.7, dtype=np.float32)
    layer.set_dense_weights("u0", "u1", dense)

    mask = layer.potential_mask("u0", "u1")
    w = layer.weight_matrix("u0", "u1")
    # In-pool entries should match dense (0.7); out-of-pool must remain 0.0.
    np.testing.assert_allclose(w[mask], 0.7)
    np.testing.assert_array_equal(w[~mask], 0.0)


def test_set_dense_weights_rejects_wrong_shape() -> None:
    layer, _, _ = _two_unit_layer()
    with pytest.raises(ValueError):
        layer.set_dense_weights("u0", "u1", np.zeros((N - 1, N), dtype=np.float32))


def test_reset_weights_resamples_data_preserves_pool() -> None:
    """reset_weights() must re-sample data but keep the potential pool topology."""
    layer, _, _ = _two_unit_layer()
    indptr_before, indices_before, data_before, _ = layer.edge_arrays("u0", "u1")
    indptr_snap = indptr_before.copy()
    indices_snap = indices_before.copy()
    data_snap = data_before.copy()

    layer.reset_weights()

    indptr_after, indices_after, data_after, _ = layer.edge_arrays("u0", "u1")
    # Topology unchanged
    np.testing.assert_array_equal(indptr_after, indptr_snap)
    np.testing.assert_array_equal(indices_after, indices_snap)
    # Data re-sampled (extremely unlikely to be bit-identical for nnz > 1)
    assert not np.array_equal(data_after, data_snap)
    # New values still within initialisation range
    assert np.all(data_after >= 0.0)
    assert np.all(data_after < 0.1 + 1e-6)


def test_csr_storage_uses_less_memory_than_dense_at_low_sparsity() -> None:
    """At lateral_sparsity=0.02, CSR must use far less memory than dense (n=512).

    Sanity check on the memory-saving claim of issue #25.
    """
    n = 512
    cfg = HeterarchicalConfig(lateral_lr=0.1, lateral_decay=0.01, lateral_sparsity=0.02)
    layer = HeterarchicalLayer(n_columns=n, config=cfg, rng=np.random.default_rng(0))
    layer.register_unit("u0")
    layer.register_unit("u1")
    layer.add_connection("u0", "u1")

    indptr, indices, data, row_idx = layer.edge_arrays("u0", "u1")
    csr_bytes = indptr.nbytes + indices.nbytes + data.nbytes + row_idx.nbytes
    dense_bytes = n * n * (4 + 1)  # float32 weight + bool mask

    # CSR should be < 25% of dense at 2% sparsity (lots of overhead margin).
    assert csr_bytes < dense_bytes * 0.25, (
        f"CSR {csr_bytes} bytes vs dense {dense_bytes} bytes — ratio {csr_bytes/dense_bytes:.3f}"
    )


def test_learn_outside_pool_stays_zero_after_many_steps() -> None:
    """After many learn() calls, weight_matrix() must remain zero outside the pool."""
    layer = _layer(sparsity=0.2, seed=33)
    layer.register_unit("u0")
    layer.register_unit("u1")
    layer.add_connection("u0", "u1")
    mask = layer.potential_mask("u0", "u1")
    assert (~mask).any()

    rng = np.random.default_rng(0)
    for _ in range(15):
        a = list(rng.choice(N, size=10, replace=False))
        b = list(rng.choice(N, size=10, replace=False))
        layer.learn([_sdr(a, uid="u0"), _sdr(b, uid="u1")])

    w = layer.weight_matrix("u0", "u1")
    assert np.all(w[~mask] == 0.0), "Out-of-pool entries became non-zero after learning"
