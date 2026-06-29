"""RandomNoiseTask (issue #32): the trivial concrete task for harness wiring.

This formalises the ad-hoc ``_random_input`` helper already living in
``halo/orchestration/pipeline.py``.  It emits unstructured Gaussian-noise
vectors with **no learnable temporal structure**, so it is the canonical
negative control: a correct HALO run on this task should see reliability
collapse toward the floor (random noise has nothing for Temporal Memory to
predict).  Its job is to prove the harness plumbing works end-to-end before
the structured Phase-2 tasks (#2/#3/#4) arrive.
"""

from __future__ import annotations

import numpy as np

from features.tasks.base import EncoderHint, Task, TaskInput, TaskLabel
from features.tasks.config import TaskConfig
from features.tasks.registry import register_task

__all__ = ["RandomNoiseTask"]


@register_task
class RandomNoiseTask(Task):
    """Emit i.i.d. Gaussian-noise vectors with no target (unsupervised control).

    Parameters
    ----------
    config:
        ``TaskConfig`` with ``type="random_noise"``.  Uses
        ``config.n_input_dim`` as the vector width.  ``encoder_hint`` is
        forced to ``"raw"`` (the task already emits a numeric vector).
    """

    name = "random_noise"

    def __init__(self, config: TaskConfig) -> None:
        self._config = config
        self._n_input_dim = config.n_input_dim
        self._step = 0
        # Set in reset(); typed as Optional so we can guard against use-before-reset.
        self._rng: np.random.Generator | None = None

    # -- Task interface ------------------------------------------------------

    def reset(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)
        self._step = 0

    def next_step(self) -> tuple[TaskInput, TaskLabel]:
        if self._rng is None:
            raise RuntimeError("next_step() called before reset()")
        raw: np.ndarray = self._rng.standard_normal(self._n_input_dim)
        label = TaskLabel(step=self._step, target=None, context_id=None)
        self._step += 1
        return raw, label

    @property
    def n_input_dim(self) -> int:
        return self._n_input_dim

    @property
    def encoder_hint(self) -> EncoderHint:
        return "raw"