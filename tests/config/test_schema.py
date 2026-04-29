"""Tests for halo.config.schema.CorticalConfig (issue #15)."""

from __future__ import annotations

from pathlib import Path

import pytest

from halo.config.schema import CorticalConfig
from halo.config.loader import load_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_kwargs() -> dict:
    return dict(
        n_columns=2048,
        sparsity=0.02,
        potential_radius=-1,
        potential_pct=0.5,
        syn_perm_connected=0.5,
        syn_perm_active_inc=0.1,
        syn_perm_inactive_dec=0.01,
        syn_perm_max=1.0,
        stimulus_threshold=0.0,
        local_area_density=0.02,
        global_inhibition=True,
        max_boost=10.0,
        duty_cycle_period=1000,
        min_pct_overlap_duty_cycles=0.001,
        min_pct_active_duty_cycles=0.001,
        update_period=50,
        learning_rate=0.1,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_valid_config_constructs() -> None:
    cfg = CorticalConfig(**_valid_kwargs())
    assert cfg.n_columns == 2048
    assert cfg.sparsity == 0.02


def test_auto_derived_trim_threshold() -> None:
    cfg = CorticalConfig(**_valid_kwargs())
    assert cfg.syn_perm_trim_threshold == pytest.approx(0.05)  # active_inc / 2


def test_auto_derived_below_stimulus_inc() -> None:
    cfg = CorticalConfig(**_valid_kwargs())
    assert cfg.syn_perm_below_stimulus_inc == pytest.approx(0.05)  # connected / 10


def test_auto_derived_updates_with_different_params() -> None:
    kw = _valid_kwargs()
    kw["syn_perm_active_inc"] = 0.2
    kw["syn_perm_connected"] = 0.6
    cfg = CorticalConfig(**kw)
    assert cfg.syn_perm_trim_threshold == pytest.approx(0.1)    # 0.2 / 2
    assert cfg.syn_perm_below_stimulus_inc == pytest.approx(0.06)  # 0.6 / 10


def test_global_inhibition_false_accepted() -> None:
    kw = _valid_kwargs()
    kw["global_inhibition"] = False
    cfg = CorticalConfig(**kw)
    assert cfg.global_inhibition is False


# ---------------------------------------------------------------------------
# Validation — permanence
# ---------------------------------------------------------------------------

def test_inactive_dec_must_be_less_than_active_inc() -> None:
    kw = _valid_kwargs()
    kw["syn_perm_inactive_dec"] = 0.1   # equal → invalid
    with pytest.raises(ValueError, match="syn_perm_inactive_dec"):
        CorticalConfig(**kw)


def test_inactive_dec_greater_raises() -> None:
    kw = _valid_kwargs()
    kw["syn_perm_inactive_dec"] = 0.2   # greater → invalid
    with pytest.raises(ValueError, match="syn_perm_inactive_dec"):
        CorticalConfig(**kw)


def test_syn_perm_max_must_be_1() -> None:
    kw = _valid_kwargs()
    kw["syn_perm_max"] = 0.9
    with pytest.raises(ValueError, match="syn_perm_max"):
        CorticalConfig(**kw)


def test_syn_perm_connected_zero_raises() -> None:
    kw = _valid_kwargs()
    kw["syn_perm_connected"] = 0.0
    with pytest.raises(ValueError, match="syn_perm_connected"):
        CorticalConfig(**kw)


def test_syn_perm_connected_one_raises() -> None:
    kw = _valid_kwargs()
    kw["syn_perm_connected"] = 1.0
    with pytest.raises(ValueError, match="syn_perm_connected"):
        CorticalConfig(**kw)


# ---------------------------------------------------------------------------
# Validation — topology
# ---------------------------------------------------------------------------

def test_n_columns_zero_raises() -> None:
    kw = _valid_kwargs()
    kw["n_columns"] = 0
    with pytest.raises(ValueError, match="n_columns"):
        CorticalConfig(**kw)


def test_sparsity_zero_raises() -> None:
    kw = _valid_kwargs()
    kw["sparsity"] = 0.0
    with pytest.raises(ValueError, match="sparsity"):
        CorticalConfig(**kw)


def test_sparsity_one_raises() -> None:
    kw = _valid_kwargs()
    kw["sparsity"] = 1.0
    with pytest.raises(ValueError, match="sparsity"):
        CorticalConfig(**kw)


def test_potential_pct_zero_raises() -> None:
    kw = _valid_kwargs()
    kw["potential_pct"] = 0.0
    with pytest.raises(ValueError, match="potential_pct"):
        CorticalConfig(**kw)


def test_potential_radius_minus_two_raises() -> None:
    kw = _valid_kwargs()
    kw["potential_radius"] = -2
    with pytest.raises(ValueError, match="potential_radius"):
        CorticalConfig(**kw)


# ---------------------------------------------------------------------------
# Validation — inhibition / boosting
# ---------------------------------------------------------------------------

def test_stimulus_threshold_negative_raises() -> None:
    kw = _valid_kwargs()
    kw["stimulus_threshold"] = -1.0
    with pytest.raises(ValueError, match="stimulus_threshold"):
        CorticalConfig(**kw)


def test_local_area_density_zero_raises() -> None:
    kw = _valid_kwargs()
    kw["local_area_density"] = 0.0
    with pytest.raises(ValueError, match="local_area_density"):
        CorticalConfig(**kw)


def test_max_boost_below_one_raises() -> None:
    kw = _valid_kwargs()
    kw["max_boost"] = 0.5
    with pytest.raises(ValueError, match="max_boost"):
        CorticalConfig(**kw)


def test_duty_cycle_period_zero_raises() -> None:
    kw = _valid_kwargs()
    kw["duty_cycle_period"] = 0
    with pytest.raises(ValueError, match="duty_cycle_period"):
        CorticalConfig(**kw)


def test_update_period_zero_raises() -> None:
    kw = _valid_kwargs()
    kw["update_period"] = 0
    with pytest.raises(ValueError, match="update_period"):
        CorticalConfig(**kw)


# ---------------------------------------------------------------------------
# Integration — baseline.yaml loads cleanly
# ---------------------------------------------------------------------------

def test_baseline_yaml_loads() -> None:
    config_path = Path(__file__).parents[2] / "configs" / "baseline.yaml"
    cfg = load_config(config_path)
    assert cfg.cortical.n_columns == 2048
    assert cfg.cortical.syn_perm_connected == pytest.approx(0.5)
    assert cfg.cortical.syn_perm_trim_threshold == pytest.approx(0.05)
    assert cfg.cortical.syn_perm_below_stimulus_inc == pytest.approx(0.05)
    assert cfg.cortical.global_inhibition is True
    # TM fields present with expected defaults
    assert cfg.cortical.cells_per_column == 32
    assert cfg.cortical.activation_threshold == 13
    assert cfg.cortical.initial_permanence == pytest.approx(0.21)


# ---------------------------------------------------------------------------
# Validation — Temporal Memory fields
# ---------------------------------------------------------------------------

def test_tm_cells_per_column_zero_raises() -> None:
    kw = _valid_kwargs()
    kw["cells_per_column"] = 0
    with pytest.raises(ValueError, match="cells_per_column"):
        CorticalConfig(**kw)


def test_tm_activation_threshold_below_min_threshold_raises() -> None:
    kw = _valid_kwargs()
    kw["activation_threshold"] = 5
    kw["min_threshold"] = 10
    with pytest.raises(ValueError, match="activation_threshold"):
        CorticalConfig(**kw)


def test_tm_max_new_synapse_count_zero_raises() -> None:
    kw = _valid_kwargs()
    kw["max_new_synapse_count"] = 0
    with pytest.raises(ValueError, match="max_new_synapse_count"):
        CorticalConfig(**kw)


def test_tm_initial_permanence_above_connected_raises() -> None:
    kw = _valid_kwargs()
    kw["initial_permanence"] = 0.6   # > syn_perm_connected (0.5)
    with pytest.raises(ValueError, match="initial_permanence"):
        CorticalConfig(**kw)


def test_tm_permanence_increment_negative_raises() -> None:
    kw = _valid_kwargs()
    kw["permanence_increment"] = -0.1
    with pytest.raises(ValueError, match="permanence_increment"):
        CorticalConfig(**kw)


def test_tm_predicted_segment_decrement_negative_raises() -> None:
    kw = _valid_kwargs()
    kw["predicted_segment_decrement"] = -0.01
    with pytest.raises(ValueError, match="predicted_segment_decrement"):
        CorticalConfig(**kw)
