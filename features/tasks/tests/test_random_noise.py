"""Tests for RandomNoiseTask (issue #32)."""

from __future__ import annotations

import numpy as np
import pytest

from features.tasks.config import TaskConfig
from features.tasks.random_noise import RandomNoiseTask


def _make_task(n_input_dim: int = 16) -> RandomNoiseTask:
    cfg = TaskConfig(type="random_noise", n_input_dim=n_input_dim, encoder_hint="raw")
    task = RandomNoiseTask(cfg)
    task.reset(seed=0)
    return task


def test_next_step_before_reset_raises() -> None:
    cfg = TaskConfig(type="random_noise", n_input_dim=4, encoder_hint="raw")
    task = RandomNoiseTask(cfg)
    with pytest.raises(RuntimeError):
        task.next_step()


def test_emits_array_of_correct_shape() -> None:
    task = _make_task(n_input_dim=16)
    raw, _label = task.next_step()
    assert isinstance(raw, np.ndarray)
    assert raw.shape == (16,)


def test_label_has_no_target() -> None:
    """Random noise is unsupervised: target must be None."""
    task = _make_task()
    _raw, label = task.next_step()
    assert label.target is None
    assert label.context_id is None


def test_step_counter_advances() -> None:
    task = _make_task()
    _r0, l0 = task.next_step()
    _r1, l1 = task.next_step()
    assert l0.step == 0
    assert l1.step == 1


def test_same_seed_reproduces_stream() -> None:
    """Determinism: identical seed -> identical sequence of inputs."""
    a = _make_task()
    b = _make_task()
    for _ in range(5):
        ra, _ = a.next_step()
        rb, _ = b.next_step()
        assert np.array_equal(ra, rb)


def test_different_seed_diverges() -> None:
    a = _make_task()
    b = RandomNoiseTask(
        TaskConfig(type="random_noise", n_input_dim=16, encoder_hint="raw")
    )
    b.reset(seed=999)
    ra, _ = a.next_step()
    rb, _ = b.next_step()
    assert not np.array_equal(ra, rb)


def test_reset_rewinds_stream() -> None:
    task = _make_task()
    first, _ = task.next_step()
    task.reset(seed=0)
    again, _ = task.next_step()
    assert np.array_equal(first, again)


def test_encoder_hint_is_raw() -> None:
    assert _make_task().encoder_hint == "raw"