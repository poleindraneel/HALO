"""HALO Phase-2 task suite (issue #32).

Importing this package registers all built-in concrete tasks (importing
``random_noise`` triggers its ``@register_task`` decorator), so callers can
``build_task`` by name without importing concrete classes themselves.
"""

from __future__ import annotations

from features.tasks.base import EncoderHint, Task, TaskInput, TaskLabel
from features.tasks.config import TaskConfig
from features.tasks.registry import (
    available_tasks,
    build_task,
    get_task_class,
    register_task,
)

# Import concrete tasks for their registration side-effects.
from features.tasks.random_noise import RandomNoiseTask  # noqa: E402,F401
from features.tasks.sequence import SequenceTask #noqa :E402,F401

__all__ = [
    "Task",
    "TaskLabel",
    "TaskInput",
    "EncoderHint",
    "TaskConfig",
    "register_task",
    "build_task",
    "get_task_class",
    "available_tasks",
    "RandomNoiseTask",
    "SequenceTask",
]