"""Tests for the Task ABC contract and TaskLabel (issue #32)."""

from __future__ import annotations

import numpy as np
import pytest

from features.tasks.base import Task, TaskLabel


# A minimal in-test concrete task used only to exercise the ABC contract.
class _DummyTask(Task):
    name = "_dummy"

    def __init__(self, n: int = 4) -> None:
        self._n = n
        self._step = 0
        self._rng: np.random.Generator | None = None

    def reset(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)
        self._step = 0

    def next_step(self) -> tuple[float, TaskLabel]:
        assert self._rng is not None
        val = float(self._rng.random())
        label = TaskLabel(step=self._step, target=val)
        self._step += 1
        return val, label

    @property
    def n_input_dim(self) -> int:
        return self._n

    @property
    def encoder_hint(self) -> str:
        return "scalar"


def test_task_cannot_be_instantiated_directly() -> None:
    """Task is abstract; instantiating it must raise TypeError."""
    with pytest.raises(TypeError):
        Task()  # type: ignore[abstract]


def test_concrete_task_exposes_required_interface() -> None:
    task = _DummyTask(n=8)
    task.reset(seed=0)
    assert task.n_input_dim == 8
    assert task.encoder_hint == "scalar"


def test_context_id_defaults_to_none() -> None:
    """Single-context tasks inherit the None default."""
    task = _DummyTask()
    assert task.context_id(0) is None
    assert task.context_id(99) is None


def test_tasklabel_is_immutable() -> None:
    """TaskLabel is frozen; field assignment must raise."""
    label = TaskLabel(step=0, target=1.0)
    with pytest.raises((AttributeError, TypeError)):
        label.target = 2.0  # type: ignore[misc]


def test_tasklabel_defaults() -> None:
    label = TaskLabel(step=3)
    assert label.step == 3
    assert label.target is None
    assert label.context_id is None
    assert label.meta == {}


def test_tasklabel_meta_is_per_instance() -> None:
    """default_factory must give each label its own dict (no shared mutable default)."""
    a = TaskLabel(step=0)
    b = TaskLabel(step=1)
    a.meta["k"] = "v"
    assert b.meta == {}