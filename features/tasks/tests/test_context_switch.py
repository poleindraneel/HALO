"""Tests for ContextSwitchTask (issue #34, Phase-2)."""

from __future__ import annotations

import pytest

# importing the package triggers registration of built-in tasks.
import features.tasks  # noqa: F401
from features.tasks.config import TaskConfig
from features.tasks.context_switch import ContextSwitchTask
from features.tasks.registry import build_task
from halo.encoders.category import CategoryEncoder

_C2 = [["A", "B", "C"], ["X", "Y", "Z"]]  # two disjoint 3-symbol contexts (union K=6)


def _union_size(contexts: list[list[str]]) -> int:
    seen: list[str] = []
    for c in contexts:
        for s in c:
            if s not in seen:
                seen.append(s)
    return len(seen)


def _cfg(contexts, *, w: int = 5, seed: int = 0, **params) -> TaskConfig:
    k = _union_size(contexts)
    return TaskConfig(
        type="context_switch",
        n_input_dim=k * w,
        encoder_hint="category",
        seed=seed,
        params={"contexts": contexts, **params},
    )


def _make(contexts, *, w: int = 5, seed: int = 0, **params) -> ContextSwitchTask:
    task = ContextSwitchTask(_cfg(contexts, w=w, seed=seed, **params))
    task.reset(seed=seed)
    return task


def _pull(task: ContextSwitchTask, n: int) -> list:
    return [task.next_step() for _ in range(n)]


# ---------------- Guards / protocols ----------------
def test_next_step_before_reset_raises() -> None:
    task = ContextSwitchTask(_cfg(_C2))
    with pytest.raises(RuntimeError):
        task.next_step()


def test_step_counter_advances() -> None:
    task = _make(_C2, switch_policy="fixed", dwell=3, n_blocks=4)
    (_r0, l0), (_r1, l1) = _pull(task, 2)
    assert l0.step == 0 and l1.step == 1


def test_registered_and_buildable() -> None:
    task = build_task(_cfg(_C2, switch_policy="fixed", dwell=3, n_blocks=4))
    assert isinstance(task, ContextSwitchTask)
    assert task.encoder_hint == "category"


def test_pairs_with_category_encoder() -> None:
    task = _make(_C2, switch_policy="fixed", dwell=3, n_blocks=4)
    enc = CategoryEncoder(n=task.n_input_dim, w=task.w, categories=task.categories)
    for raw, _lbl in _pull(task, 6):
        assert isinstance(raw, str)
        assert len(enc.encode(raw).active_indices) == task.w


# ---------------- Acceptance 1: fixed-dwell context-label sequence ----------------
def test_fixed_dwell_context_sequence() -> None:
    task = _make(_C2, switch_policy="fixed", dwell=3, n_blocks=4)
    assert task.context_stream == [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]
    # labels agree with the stream and with the context_id() override
    for raw, lbl in _pull(task, task.total_steps):
        assert lbl.context_id == task.context_id(lbl.step)


def test_raw_symbol_never_leaks_context() -> None:
    # every clean observation is a plain union-alphabet symbol; nothing tags context
    task = _make(_C2, switch_policy="fixed", dwell=3, n_blocks=4)
    for raw, _lbl in _pull(task, task.total_steps):
        assert raw in task.categories


# ---------------- Acceptance 2: per-context next-symbol labels well-formed ----------------
def test_per_context_targets_are_clean_next_symbol() -> None:
    task = _make(_C2, switch_policy="fixed", dwell=3, n_blocks=4)
    clean = task.clean_stream
    total = task.total_steps
    for i, (_raw, lbl) in enumerate(_pull(task, total)):
        assert lbl.target == clean[(i + 1) % total]
        assert lbl.target in task.categories


def test_resume_pointer_not_restart() -> None:
    # dwell (2) not a multiple of context length (3): context 0 must RESUME (C,A), not restart (A,B)
    task = _make(_C2, switch_policy="fixed", dwell=2, n_blocks=4)
    # blocks: [A,B]0 [X,Y]1 [C,A]0 [Z,X]1
    assert task.clean_stream == ["A", "B", "X", "Y", "C", "A", "Z", "X"]


# ---------------- Acceptance 3: aggregate context proportions ----------------
def test_fixed_dwell_proportions_balanced() -> None:
    task = _make(_C2, switch_policy="fixed", dwell=4, n_blocks=8)
    stream = task.context_stream
    assert stream.count(0) == stream.count(1)  # equal dwell + even blocks -> exactly balanced


def test_geometric_proportions_within_tolerance() -> None:
    task = _make(_C2, switch_policy="geometric", p_switch=0.2, n_blocks=4000, seed=11)
    stream = task.context_stream
    frac0 = stream.count(0) / len(stream)
    assert abs(frac0 - 0.5) < 0.03


# ---------------- Acceptance 4: empirical noise / jitter rates ----------------
@pytest.mark.parametrize("p", [0.0, 0.15, 0.4])
def test_symbol_noise_rate_within_tolerance(p: float) -> None:
    task = _make(_C2, switch_policy="fixed", dwell=6, n_blocks=4,
                 p_symbol_noise=p, seed=7)
    steps = _pull(task, 20000)  # cycles past the schedule; fresh noise each pass
    rate = sum(lbl.meta["noised"] for _, lbl in steps) / len(steps)
    assert abs(rate - p) < 0.02


@pytest.mark.parametrize("p", [0.0, 0.3, 0.7])
def test_boundary_jitter_rate_within_tolerance(p: float) -> None:
    task = _make(_C2, switch_policy="fixed", dwell=4, n_blocks=4000,
                 p_boundary_jitter=p, seed=5)
    steps = _pull(task, task.total_steps)
    jittered = sum(lbl.meta["boundary_jitter"] for _, lbl in steps)
    rate = jittered / len(task.boundaries)
    assert abs(rate - p) < 0.03


def test_zero_noise_never_substitutes() -> None:
    task = _make(_C2, switch_policy="fixed", dwell=5, n_blocks=6, p_symbol_noise=0.0)
    assert not any(lbl.meta["noised"] for _, lbl in _pull(task, task.total_steps))


def test_zero_jitter_never_leaks() -> None:
    task = _make(_C2, switch_policy="fixed", dwell=5, n_blocks=6, p_boundary_jitter=0.0)
    assert not any(lbl.meta["boundary_jitter"] for _, lbl in _pull(task, task.total_steps))


def test_noise_substitutes_to_different_symbol() -> None:
    task = _make(_C2, switch_policy="fixed", dwell=6, n_blocks=6,
                 p_symbol_noise=0.6, seed=3)
    for raw, lbl in _pull(task, task.total_steps):
        if lbl.meta["noised"]:
            assert raw != lbl.meta["clean_symbol"]


def test_jitter_is_observation_only() -> None:
    # at a jittered step: obs == incoming context's clean symbol, but context_id/target stay clean.
    # p_boundary_jitter=1.0 -> every boundary jitters; p_symbol_noise=0 -> obs == observed_base.
    task = _make(_C2, switch_policy="fixed", dwell=3, n_blocks=8,
                 p_boundary_jitter=1.0, seed=1)
    clean = task.clean_stream
    ctx = task.context_stream
    steps = _pull(task, task.total_steps)
    for b in task.boundaries:
        raw, lbl = steps[b - 1]              # the jittered step is the outgoing block's last step
        assert lbl.meta["boundary_jitter"] is True
        assert raw == clean[b]               # incoming context's clean symbol leaked one step early
        assert lbl.context_id == ctx[b - 1]  # context label unchanged (still outgoing)
        assert lbl.context_id != ctx[b]
        assert lbl.target == clean[b]        # target stays the clean incoming symbol


# ---------------- Determinism ----------------
def test_same_seed_reproduces_stream() -> None:
    a = _make(_C2, switch_policy="geometric", p_switch=0.3, n_blocks=30,
              p_symbol_noise=0.3, p_boundary_jitter=0.5, seed=42)
    b = _make(_C2, switch_policy="geometric", p_switch=0.3, n_blocks=30,
              p_symbol_noise=0.3, p_boundary_jitter=0.5, seed=42)
    sa = [(r, l.target, l.context_id, l.meta["noised"]) for r, l in _pull(a, 100)]
    sb = [(r, l.target, l.context_id, l.meta["noised"]) for r, l in _pull(b, 100)]
    assert sa == sb


def test_reset_rewinds_stream() -> None:
    task = _make(_C2, switch_policy="geometric", p_switch=0.3, n_blocks=30,
                 p_symbol_noise=0.4, seed=9)
    first = [r for r, _ in _pull(task, 60)]
    task.reset(seed=9)
    second = [r for r, _ in _pull(task, 60)]
    assert first == second


def test_different_seed_diverges() -> None:
    a = _make(_C2, switch_policy="geometric", p_switch=0.3, n_blocks=30, seed=1)
    b = _make(_C2, switch_policy="geometric", p_switch=0.3, n_blocks=30, seed=2)
    sa = [r for r, _ in _pull(a, 100)]
    sb = [r for r, _ in _pull(b, 100)]
    assert sa != sb


# ---------------- Validation rejections ----------------
@pytest.mark.parametrize(
    "params, nid",
    [
        ({}, 30),                                                      # missing contexts
        ({"contexts": [["A", "B"]]}, 10),                              # <2 contexts
        ({"contexts": [["A"], []]}, 5),                                # empty context
        ({"contexts": [["A"], ["X", 1]]}, 10),                         # non-str entry
        ({"contexts": [["A"], ["X", True]]}, 10),                      # bool entry
        ({"contexts": [["A"], "XY"]}, 15),                             # string sub-seq
        ({"contexts": "ABC"}, 15),                                     # string contexts
        ({"contexts": _C2, "switch_policy": "rng"}, 30),               # bad policy
        ({"contexts": _C2, "switch_policy": "fixed", "dwell": 0}, 30), # dwell < 1
        ({"contexts": _C2, "p_switch": 0.0}, 30),                      # p_switch <= 0
        ({"contexts": _C2, "p_switch": 1.5}, 30),                      # p_switch > 1
        ({"contexts": _C2, "dwell_min": 9, "dwell_max": 3}, 30),       # max < min
        ({"contexts": _C2, "n_blocks": 0}, 30),                        # n_blocks < 1
        ({"contexts": _C2, "p_symbol_noise": 2.0}, 30),                # noise out of range
        ({"contexts": _C2, "p_boundary_jitter": 0.3, "n_blocks": 1}, 30),  # jitter needs boundary
        ({"contexts": _C2}, 7),                                        # n_input_dim % K != 0
    ],
)
def test_invalid_params_rejected(params, nid) -> None:
    cfg = TaskConfig(type="context_switch", n_input_dim=nid,
                     encoder_hint="category", seed=0, params=params)
    with pytest.raises((ValueError, TypeError)):
        ContextSwitchTask(cfg)