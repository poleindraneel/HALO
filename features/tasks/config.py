"""Staged ``TaskConfig`` (issue #32).

This is the staging draft of the config object that will eventually be merged
into ``halo/config/schema.py`` by the maintainer.  It is kept here in
``features/`` so it can be developed and tested without touching the shared
``halo/`` package yet.

Design note — why a generic config instead of a per-task union
--------------------------------------------------------------
The issue *suggests* a ``TaskConfig`` union (one dataclass per task, like the
``ScalarEncoderConfig | CategoryEncoderConfig`` encoder pattern).  We instead
use a single generic config carrying ``type`` + a ``params`` dict, because the
issue's stated goal is that tasks are *"pluggable via config — never
hard-imported."*  A rigid union would force an edit to ``schema.py`` for every
new task (#2/#3/#4), which is the opposite of pluggable.  With this design,
adding a task means registering a class (see ``registry.py``); ``schema.py``
never changes again.  Per-task validation lives in each Task's ``reset``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["TaskConfig"]

_VALID_ENCODER_HINTS = {"scalar", "category", "raw"}


@dataclass
class TaskConfig:
    """Configuration selecting and parameterising a concrete :class:`Task`.

    Parameters
    ----------
    type:
        Registry key of the concrete task (e.g. ``"random_noise"``).
    n_input_dim:
        Input width the task produces; must match ``HALOConfig.n_input_dim``.
    encoder_hint:
        ``"scalar"``, ``"category"``, or ``"raw"`` — how the harness should
        encode this task's raw inputs.
    seed:
        Seed passed to :meth:`Task.reset` for a reproducible stream.
    params:
        Task-specific keyword arguments (validated by the concrete task).
    """

    type: str
    n_input_dim: int
    encoder_hint: str = "raw"
    seed: int = 0
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.type:
            raise ValueError("task.type must be a non-empty registry key")
        if self.n_input_dim < 1:
            raise ValueError(
                f"task.n_input_dim must be ≥ 1, got {self.n_input_dim}"
            )
        if self.encoder_hint not in _VALID_ENCODER_HINTS:
            raise ValueError(
                f"task.encoder_hint must be one of {sorted(_VALID_ENCODER_HINTS)}, "
                f"got {self.encoder_hint!r}"
            )