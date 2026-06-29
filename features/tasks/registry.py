"""Task registry (issue #32): pluggable tasks selected by config, not imports.

Concrete tasks register themselves with the :func:`register_task` decorator;
the harness builds a task from a :class:`~features.tasks.config.TaskConfig`
via :func:`build_task`.  The pipeline therefore never hard-imports a concrete
task class.

Example
-------
>>> from features.tasks.config import TaskConfig
>>> task = build_task(TaskConfig(type="random_noise", n_input_dim=64,
...                              encoder_hint="raw"))
>>> task.reset(seed=0)
>>> raw, label = task.next_step()
"""

from __future__ import annotations

import logging
from typing import Callable, TypeVar

from features.tasks.base import Task
from features.tasks.config import TaskConfig

logger = logging.getLogger(__name__)

__all__ = ["register_task", "build_task", "get_task_class", "available_tasks"]

# Maps registry key -> concrete Task subclass.
_REGISTRY: dict[str, type[Task]] = {}

T = TypeVar("T", bound=Task)


def register_task(cls: type[T]) -> type[T]:
    """Class decorator: register a concrete :class:`Task` under its ``name``.

    Raises
    ------
    ValueError
        If ``cls.name`` is empty or already registered.
    """
    key = cls.name
    if not key:
        raise ValueError(
            f"{cls.__name__} must set a non-empty class attribute `name` to register"
        )
    if key in _REGISTRY and _REGISTRY[key] is not cls:
        raise ValueError(f"task name {key!r} already registered to {_REGISTRY[key].__name__}")
    _REGISTRY[key] = cls
    logger.debug("Registered task %r -> %s", key, cls.__name__)
    return cls


def get_task_class(name: str) -> type[Task]:
    """Return the registered task class for *name*.

    Raises
    ------
    KeyError
        If no task is registered under *name*.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown task type {name!r}; registered: {sorted(_REGISTRY)}"
        ) from None


def available_tasks() -> list[str]:
    """Return the sorted list of registered task names."""
    return sorted(_REGISTRY)


def build_task(config: TaskConfig) -> Task:
    """Instantiate the concrete task named by ``config.type``.

    The task is constructed with ``config`` and then ``reset(config.seed)`` is
    called so it is ready to stream immediately.

    Parameters
    ----------
    config:
        Validated :class:`TaskConfig`.

    Returns
    -------
    Task
        A reset, ready-to-stream task instance.
    """
    cls = get_task_class(config.type)
    task = cls(config)
    task.reset(config.seed)
    logger.info("Built task %r (n_input_dim=%d)", config.type, config.n_input_dim)
    return task