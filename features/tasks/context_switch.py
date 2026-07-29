"""ContextSwitchTask (issue #34, Phase-2): interleave >=2 latent sub-sequences.

Extends the single-context SequenceTask (#33) to a *multi-context* stream:
the generator runs one sub-sequence for a dwell, switches to another, and
repeats. Each step's TaskLabel carries the CLEAN next symbol as `target` and
the active `context_id` (0-based) - but the raw emitted symbol never encodes
the context, so HALO must INFER the active context from temporal structure
alone. This is the core benchmark feeding Phase 3 (multi-context consensus).

Two independent noise channels:
* `p_symbol_noise`   - substitution on the observation only (as in #33);
                       `target`/`context_id` stay clean.
* `p_boundary_jitter`- at a context switch, the incoming context's first symbol
                       leaks one step early into the outgoing context's last
                       step (sensor delay / percept overlap). Observation-only:
                       the step's `context_id`/`target` remain the clean value.

encoder_hint="category"; the union alphabet across all contexts sizes the
CategoryEncoder (n_input_dim = K * w, K = |union alphabet|).
"""
from __future__ import annotations

import numpy as np

from features.tasks.base import EncoderHint, Task, TaskInput, TaskLabel
from features.tasks.config import TaskConfig
from features.tasks.registry import register_task

__all__ = ["ContextSwitchTask"]


def _build_union_alphabet(
    contexts: object,
) -> tuple[list[list[str]], list[str], dict[str, int]]:
    """Validate `contexts` and derive the shared union alphabet.

    Returns
    -------
    (contexts, union_alphabet, idx_of)
        `contexts`        - the validated sub-sequences as lists of str.
        `union_alphabet`  - every distinct symbol across all contexts, in
                            first-appearance order (deterministic; sizes the
                            CategoryEncoder as K = len(union_alphabet)).
        `idx_of`          - symbol -> its index in `union_alphabet`.
    """
    if isinstance(contexts, (str, bytes)):
        raise TypeError("params['contexts'] must be a list of sub-sequences, not a string")
    if not isinstance(contexts, (list, tuple)):
        raise TypeError("params['contexts'] must be a list of sub-sequences")
    seqs = list(contexts)
    if len(seqs) < 2:
        raise ValueError(f"ContextSwitchTask needs >=2 contexts, got {len(seqs)}")

    out_contexts: list[list[str]] = []
    union: list[str] = []
    seen: set[str] = set()
    for c, seq in enumerate(seqs):
        if isinstance(seq, (str, bytes)):
            raise TypeError(f"context {c} must be a list of symbols, not a string")
        if not isinstance(seq, (list, tuple)):
            raise TypeError(f"context {c} must be a list/tuple of symbol strings")
        symbols = list(seq)
        if not symbols:
            raise ValueError(f"context {c} must be non-empty")
        for s in symbols:
            if isinstance(s, bool) or not isinstance(s, str):
                raise TypeError(
                    f"context {c} entries must be strings, got {type(s).__name__}"
                )
            if s not in seen:
                seen.add(s)
                union.append(s)
        out_contexts.append(symbols)

    idx_of = {s: i for i, s in enumerate(union)}
    return out_contexts, union, idx_of


_SWITCH_POLICIES = {"fixed", "geometric", "uniform"}


@register_task
class ContextSwitchTask(Task):
    """Interleave >=2 latent sub-sequences; infer-only context labels.

    Stochastic schedule (dwell timing + boundary jitter) is built in
    :meth:`reset` because it consumes the RNG; ``__init__`` only validates
    params and derives the encoder sizing (which are RNG-independent).
    """

    name = "context_switch"

    def __init__(self, config: TaskConfig):
        super().__init__(config)
        p = config.params

        if "contexts" not in p:
            raise ValueError("ContextSwitchTask requires params['contexts']")
        self._contexts, self._union, self._idx_of = _build_union_alphabet(p["contexts"])
        k = len(self._union)

        self._switch_policy: str = str(p.get("switch_policy", "geometric"))
        if self._switch_policy not in _SWITCH_POLICIES:
            raise ValueError(
                f"params['switch_policy'] must be one of {sorted(_SWITCH_POLICIES)}, "
                f"got {self._switch_policy!r}"
            )

        # --- dwell timing (only the active policy's params matter, but all are validated) ---
        self._dwell: int = int(p.get("dwell", max(len(c) for c in self._contexts)))
        if self._dwell < 1:
            raise ValueError(f"params['dwell'] must be >=1, got {self._dwell}")

        self._p_switch: float = float(p.get("p_switch", 0.1))
        if not 0.0 < self._p_switch <= 1.0:
            raise ValueError(f"params['p_switch'] must be in (0,1], got {self._p_switch}")

        self._dwell_min: int = int(p.get("dwell_min", 2))
        self._dwell_max: int = int(p.get("dwell_max", 8))
        if self._dwell_min < 1:
            raise ValueError(f"params['dwell_min'] must be >=1, got {self._dwell_min}")
        if self._dwell_max < self._dwell_min:
            raise ValueError(
                f"params['dwell_max']({self._dwell_max}) must be >= "
                f"dwell_min({self._dwell_min})"
            )

        self._n_blocks: int = int(p.get("n_blocks", 8))
        if self._n_blocks < 1:
            raise ValueError(f"params['n_blocks'] must be >=1, got {self._n_blocks}")

        # --- noise channels ---
        self._p_symbol_noise: float = float(p.get("p_symbol_noise", 0.0))
        if not 0.0 <= self._p_symbol_noise <= 1.0:
            raise ValueError(
                f"params['p_symbol_noise'] must be in [0,1], got {self._p_symbol_noise}"
            )
        if self._p_symbol_noise > 0.0 and k < 2:
            raise ValueError("p_symbol_noise > 0 requires a union alphabet of size >= 2")

        self._p_boundary_jitter: float = float(p.get("p_boundary_jitter", 0.0))
        if not 0.0 <= self._p_boundary_jitter <= 1.0:
            raise ValueError(
                f"params['p_boundary_jitter'] must be in [0,1], got {self._p_boundary_jitter}"
            )
        if self._p_boundary_jitter > 0.0 and self._n_blocks < 2:
            raise ValueError("p_boundary_jitter > 0 requires n_blocks >= 2 (needs a boundary)")

        # --- encoder sizing: CategoryEncoder needs n = K * w, w >= 1 ---
        self._n_input_dim: int = config.n_input_dim
        if self._n_input_dim % k != 0:
            raise ValueError(
                f"n_input_dim({self._n_input_dim}) must be a multiple of the union "
                f"alphabet size K={k} (so w = n_input_dim // K is an integer)"
            )
        self._w: int = self._n_input_dim // k
        if self._w < 1:
            raise ValueError(
                f"n_input_dim({self._n_input_dim}) too small for K={k}; needs w >= 1"
            )

        # --- schedule state: built in reset() (needs the RNG) ---
        self._step: int = 0
        self._rng: np.random.Generator | None = None
        self._clean_stream: list[str] | None = None
        self._observed_base: list[str] | None = None
        self._context_of: list[int] | None = None
        self._targets: list[str] | None = None
        self._jittered: list[bool] | None = None
        self._boundaries: list[int] | None = None

    def _roll_dwell(self) -> int:
        """Sample one dwell length (>=1) according to ``switch_policy``.

        Uses ``self._rng`` (so only valid after :meth:`reset`).

        * ``fixed``     -> constant ``self._dwell``.
        * ``geometric`` -> ``rng.geometric(p_switch)``; support {1,2,...},
                           mean ``1/p_switch``. Memoryless: how long we've
                           dwelt gives no information about time-to-switch,
                           so the pipeline cannot time switches with a clock.
        * ``uniform``   -> ``Uniform{dwell_min .. dwell_max}`` inclusive.
        """
        assert self._rng is not None  # guaranteed by callers (built in reset)
        if self._switch_policy == "fixed":
            return self._dwell
        if self._switch_policy == "geometric":
            return int(self._rng.geometric(self._p_switch))
        # uniform: integers() upper bound is exclusive, so +1 for an inclusive max
        return int(self._rng.integers(self._dwell_min, self._dwell_max + 1))

    def reset(self, seed: int) -> None:
        """Reseed and build the full stochastic schedule (same seed -> same stream).

        Builds, over ``n_blocks`` round-robin dwell blocks:
          _clean_stream   - pure schedule symbols (resume pointers; no noise/jitter)
          _context_of      - active context id per step
          _targets         - clean next symbol, cyclic at the wrap (honest across switches)
          _observed_base   - _clean_stream with boundary jitter baked in (obs-only)
          _jittered        - per-step flag: was this obs replaced by a jitter leak
          _boundaries      - step indices where a switch begins (first step of each new block)
        Substitution noise is NOT baked here - it is rolled fresh per step in
        next_step() (as in #33), so cycling past the schedule reuses the clean
        structure but draws independent noise each pass.
        """
        self._rng = np.random.default_rng(seed)
        self._step = 0

        n_ctx = len(self._contexts)
        ptrs = [0] * n_ctx  # per-context resume pointer: advances only while that context is active
        clean: list[str] = []
        context_of: list[int] = []
        block_start: list[int] = []

        for b in range(self._n_blocks):
            ctx = b % n_ctx  # round-robin context selection
            dwell = self._roll_dwell()
            block_start.append(len(clean))
            length_c = len(self._contexts[ctx])
            for _ in range(dwell):
                clean.append(self._contexts[ctx][ptrs[ctx] % length_c])
                context_of.append(ctx)
                ptrs[ctx] += 1

        total = len(clean)
        targets = [clean[(t + 1) % total] for t in range(total)]
        boundaries = block_start[1:]  # first block has no incoming boundary

        # Boundary jitter: the incoming context's first (clean) symbol leaks one step
        # early into the OUTGOING context's last step. Observation-only: context_of and
        # target at that step stay the clean outgoing value.
        observed_base = list(clean)
        jittered = [False] * total
        if self._p_boundary_jitter > 0.0:
            for b_idx in boundaries:
                if self._rng.random() < self._p_boundary_jitter:
                    observed_base[b_idx - 1] = clean[b_idx]
                    jittered[b_idx - 1] = True

        self._clean_stream = clean
        self._observed_base = observed_base
        self._context_of = context_of
        self._targets = targets
        self._jittered = jittered
        self._boundaries = boundaries

    def next_step(self) -> tuple[TaskInput, TaskLabel]:
        if self._rng is None:
            raise RuntimeError("next_step() called before reset()")
        assert (
            self._clean_stream is not None
            and self._observed_base is not None
            and self._context_of is not None
            and self._targets is not None
            and self._jittered is not None
        )

        total = len(self._clean_stream)
        idx = self._step % total  # cycle safely past the built schedule length
        clean = self._clean_stream[idx]
        base = self._observed_base[idx]  # clean, or jitter-leaked at a boundary

        # Substitution noise on the OBSERVATION only (fresh per-step, as in #33).
        noised = self._p_symbol_noise > 0.0 and bool(self._rng.random() < self._p_symbol_noise)
        if noised:
            k = len(self._union)
            offset = int(self._rng.integers(1, k))  # 1..k-1 -> guaranteed different from `base`
            observed = self._union[(self._idx_of[base] + offset) % k]
        else:
            observed = base

        label = TaskLabel(
            step=self._step,
            target=self._targets[idx],          # clean next symbol (honest across switches)
            context_id=self._context_of[idx],   # active latent context (scoring hook)
            meta={
                "noised": noised,
                "boundary_jitter": self._jittered[idx],
                "clean_symbol": clean,
            },
        )
        self._step += 1
        return observed, label

    def context_id(self, step: int) -> int | None:
        """Latent context active at *step* (cyclic), or None before reset.

        Overrides Task.context_id so the harness can query any step without
        re-running the stream. Returns the same value the step's TaskLabel
        carries.
        """
        if self._context_of is None:
            return None
        return self._context_of[step % len(self._context_of)]

    @property
    def n_input_dim(self) -> int:
        return self._n_input_dim

    @property
    def encoder_hint(self) -> EncoderHint:
        return "category"

    # ---------------- Introspection (used by the harness/tests) ----------------
    @property
    def categories(self) -> list[str]:
        """Union alphabet across all contexts (for building the CategoryEncoder)."""
        return list(self._union)

    @property
    def w(self) -> int:
        """Active bits per category (n_input_dim // K)."""
        return self._w

    @property
    def contexts(self) -> list[list[str]]:
        """The configured sub-sequences (copy)."""
        return [list(c) for c in self._contexts]

    @property
    def n_contexts(self) -> int:
        return len(self._contexts)

    @property
    def switch_policy(self) -> str:
        return self._switch_policy

    @property
    def n_blocks(self) -> int:
        return self._n_blocks

    @property
    def p_symbol_noise(self) -> float:
        return self._p_symbol_noise

    @property
    def p_boundary_jitter(self) -> float:
        return self._p_boundary_jitter

    # --- schedule-dependent views (only valid after reset) ---
    @property
    def total_steps(self) -> int:
        """Length of one full built schedule (sum of block dwells)."""
        if self._clean_stream is None:
            raise RuntimeError("total_steps is available only after reset()")
        return len(self._clean_stream)

    @property
    def clean_stream(self) -> list[str]:
        if self._clean_stream is None:
            raise RuntimeError("clean_stream is available only after reset()")
        return list(self._clean_stream)

    @property
    def targets(self) -> list[str]:
        if self._targets is None:
            raise RuntimeError("targets is available only after reset()")
        return list(self._targets)

    @property
    def context_stream(self) -> list[int]:
        """Per-step context id for the whole built schedule."""
        if self._context_of is None:
            raise RuntimeError("context_stream is available only after reset()")
        return list(self._context_of)

    @property
    def boundaries(self) -> list[int]:
        """Step indices where a switch begins (denominator for jitter rate)."""
        if self._boundaries is None:
            raise RuntimeError("boundaries is available only after reset()")
        return list(self._boundaries)