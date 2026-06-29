"""Tests for the task registry and TaskConfig (issue #32)."""

from __future__ import annotations

import numpy as np
import pytest

# Importing the package triggers registration of built-in tasks.
import features.tasks  # noqa: F401
from features.tasks.base import Task, TaskLabel
from features.tasks.config import TaskConfig
from features.tasks.registry import (
    available_tasks,
    build_task,
    get_task_class,
    register_task,
)


# --- TaskConfig validation -------------------------------------------------

def test_config_rejects_empty_type() -> None:
    with pytest.raises(ValueError):
        TaskConfig(type="", n_input_dim=4)


def test_config_rejects_bad_input_dim() -> None:
    with pytest.raises(ValueError):
        TaskConfig(type="random_noise", n_input_dim=0)


def test_config_rejects_unknown_encoder_hint() -> None:
    with pytest.raises(ValueError):
        TaskConfig(type="random_noise", n_input_dim=4, encoder_hint="sparse")


# --- Registry --------------------------------------------------------------

def test_random_noise_is_registered() -> None:
    assert "random_noise" in available_tasks()


def test_get_task_class_returns_subclass() -> None:
    cls = get_task_class("random_noise")
    assert issubclass(cls, Task)


def test_get_unknown_task_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        get_task_class("does_not_exist")


def test_register_rejects_nameless_task() -> None:
    class _Nameless(Task):  # name left as "" -> invalid
        def reset(self, seed: int) -> None: ...
        def next_step(self): return 0.0, TaskLabel(step=0)
        @property
        def n_input_dim(self) -> int: return 1
        @property
        def encoder_hint(self) -> str: return "scalar"

    with pytest.raises(ValueError):
        register_task(_Nameless)


# --- build_task ------------------------------------------------------------

def test_build_task_returns_ready_stream() -> None:
    cfg = TaskConfig(type="random_noise", n_input_dim=32, encoder_hint="raw")
    task = build_task(cfg)
    raw, label = task.next_step()  # no manual reset needed: build_task resets
    assert isinstance(raw, np.ndarray)
    assert raw.shape == (32,)
    assert label.step == 0


def test_build_task_is_deterministic_via_config_seed() -> None:
    cfg = TaskConfig(type="random_noise", n_input_dim=8, encoder_hint="raw", seed=7)
    a = build_task(cfg)
    b = build_task(cfg)
    ra, _ = a.next_step()
    rb, _ = b.next_step()
    assert np.array_equal(ra, rb)


def test_build_unknown_task_raises() -> None:
    with pytest.raises(KeyError):
        build_task(TaskConfig(type="ghost", n_input_dim=4))