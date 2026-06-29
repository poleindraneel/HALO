"""Abstract base class for HALO experimental tasks (Phase 2 keystone, issue #32).

A :class:`Task` is a *source of inputs and ground-truth labels* for the
experimental harness.  It sits in front of the HALO pipeline: the harness
pulls one ``(raw_input, label)`` pair per timestep from a Task, feeds the
raw input through the pipeline, and scores the pipeline's output against the
label.

Design contract
---------------
* A Task **emits raw values** (``float`` for scalar streams, ``str`` for
  category streams, or a pre-built ``np.ndarray``) plus an
  :class:`TaskLabel`.  It does **not** encode to an SDR itself — encoding
  stays owned by the pipeline, which is why a Task advertises an
  ``encoder_hint`` so the harness can size/select the right encoder.
* Tasks are pluggable via config and registered by name (see
  ``features/tasks/registry.py``); they are never hard-imported by the
  pipeline.

This mirrors the existing input contract in
``halo/orchestration/pipeline.py`` (`_prepare_input` / `_random_input`),
where an input item is already one of ``np.ndarray | float | str``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from typing import TYPE_CHECKING
import numpy as np
if TYPE_CHECKING:
    from features.tasks.config import TaskConfig
__all__ = ["TaskInput", "EncoderHint", "TaskLabel", "Task"]

# A single raw input value the pipeline knows how to consume.
# (Matches halo/orchestration/pipeline.py::_prepare_input accepted types.)
TaskInput = np.ndarray | float | str

# Tells the harness which encoder family fits this task's raw inputs.
#   "scalar"   -> ScalarEncoder          (raw input is a float)
#   "category" -> CategoryEncoder        (raw input is a str)
#   "raw"      -> no encoder; input is already an np.ndarray bit-vector
EncoderHint = str


@dataclass(frozen=True)
class TaskLabel:
    """Ground-truth record emitted alongside each raw input.

    Frozen (immutable): a label is a record of fact for a single step and
    must not be mutated after creation — this prevents the harness from
    accidentally rewriting history while scoring.

    Parameters
    ----------
    step:
        Zero-based index of the timestep this label belongs to.
    target:
        The ground-truth value the pipeline output is scored against.
        ``None`` for unsupervised / no-target tasks (e.g. random noise).
    context_id:
        Which latent context generated this step, or ``None`` if the task
        is single-context.  Used by the multi-context tasks (#3/#4) to score
        context-disambiguation.
    meta:
        Free-form provenance/debug fields (never used for scoring).
    """

    step: int
    target: TaskInput | None = None
    context_id: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class Task(ABC):
    """Abstract experimental task: a stream of ``(raw_input, label)`` pairs.

    Concrete subclasses implement a single data-generating process.  The
    harness drives a Task like an iterator:

    >>> task.reset(seed=0)
    >>> raw, label = task.next_step()
    >>> # ... feed `raw` to the pipeline, score output against `label` ...

    Subclasses MUST set the class attribute :attr:`name` (used by the
    registry) and implement :meth:`reset`, :meth:`next_step`,
    :attr:`n_input_dim`, and :attr:`encoder_hint`.
    """

    #: Registry key.  Every concrete subclass overrides this.
    name: str = ""

    def __init__(self,config:TaskConfig):
        """Every task is a built from. a class:`TaskConfig`(see registry.built_task)."""
        self._config=config

    @abstractmethod
    def reset(self, seed: int) -> None:
        """(Re)initialise the task's internal state with a fixed *seed*.

        Calling ``reset`` with the same seed MUST reproduce the exact same
        stream of ``(raw_input, label)`` pairs (determinism is required so
        experiments are repeatable).

        Parameters
        ----------
        seed:
            Seed for the task's internal ``np.random.default_rng``.
        """
        ...

    @abstractmethod
    def next_step(self) -> tuple[TaskInput, TaskLabel]:
        """Produce the next ``(raw_input, label)`` pair and advance one step.

        Returns
        -------
        tuple[TaskInput, TaskLabel]
            ``raw_input`` is a ``float``, ``str``, or ``np.ndarray`` (matching
            :attr:`encoder_hint`); ``label`` is the :class:`TaskLabel` for the
            same step.

        Raises
        ------
        RuntimeError
            If called before :meth:`reset`.
        """
        ...

    @property
    @abstractmethod
    def n_input_dim(self) -> int:
        """Width the encoder must produce / the pipeline's ``n_input_dim``.

        For ``encoder_hint == "raw"`` this is the length of the emitted
        ``np.ndarray``.  For scalar/category hints it is the SDR width the
        chosen encoder will output.
        """
        ...

    @property
    @abstractmethod
    def encoder_hint(self) -> EncoderHint:
        """One of ``"scalar"``, ``"category"``, or ``"raw"`` (see module docs)."""
        ...

    def context_id(self, step: int) -> int | None:
        """Latent context active at *step*, or ``None`` for single-context tasks.

        Default implementation returns ``None``.  Multi-context tasks
        (#3/#4) override this.  Provided as a queryable method (separate from
        the per-step label) so the harness can ask about any step without
        re-running the stream.
        """
        return None