""" Tests for SequenceTask (issue #33, Phase-2)"""

from __future__ import annotations
import pytest
# importing the package triggers registration of built-in tasks.
import features.tasks #noqa: F401
from features.tasks.base import Task,TaskLabel
from features.tasks.config import TaskConfig
from features.tasks.registry import build_task
from features.tasks.sequence import SequenceTask
from halo.encoders.category import CategoryEncoder

def _cfg(alphabet,*,w:int=5,seed:int=0,**params)->TaskConfig:
    k=alphabet if isinstance(alphabet,int) else len(alphabet)
    return TaskConfig(
        type="sequence",
        n_input_dim=k*w,
        encoder_hint="category",
        seed=seed,
        params={"alphabet":alphabet,**params},
    )

def _make(alphabet,*,w:int=5,seed:int=0,**params)->SequenceTask:
    task=SequenceTask(_cfg(alphabet,w=w,seed=seed,**params))
    task.reset(seed=seed)
    return task
def _pull(task:SequenceTask,n:int)->list:
    return [task.next_step() for _ in range(n)]

# Guard/protocols
def test_next_step_before_reset_raises()->None:
    task=SequenceTask(_cfg(4))
    with pytest.raises(RuntimeError):
        task.next_step()
def test_step_counter_advances()->None:
    task=_make(4)
    (_r0,l0),(_r1,l1)=_pull(task,2)
    assert l0.step==0 and l1.step==1

# Acceptance 1: noise-free k=4 -> expected next-symbol labels

def test_noise_free_k4_expeected_labels()->None:
    task=_make(4,length=4,n_repetitions=3)
    steps=_pull(task,12)
    values=[raw for raw,_ in steps]
    targets=[lbl.target for _,lbl in steps]
    assert values == ["A","B","C","D"]*3
    assert targets==["B","C","D","A"]*3
    assert not any(lbl.meta["noised"]for _,lbl in steps)

def test_length_wraps_beyond_alphabet()->None:
    task=_make(3,length=5,n_repetitions=1)
    assert task.clean_stream == ["A", "B", "C", "A", "B"]
    assert task.targets == ["B", "C", "A", "B", "A"]
def test_explicit_string_alphabet()->None:
    task=_make(["red","green","blue"],w=2,n_repetitions=2)
    assert task.categories==['red','green','blue']
    assert task.clean_stream==['red',"green","blue","red","green","blue"]

def test_stream_cycles_past_designed_length()->None:
    task=_make(4,length=4,n_repetitions=1)
    values=[raw for raw,_ in _pull(task,6)] #pull past total_steps=4.
    assert values== ["A", "B", "C", "D", "A", "B"]

# ---Acceptance 2: noise rate matches p_noise within tolerace

# --- Acceptance 2: noise rate matches p_noise within tolerance --------------

@pytest.mark.parametrize("p_noise", [0.1, 0.25, 0.5])
def test_noise_rate_within_tolerance(p_noise: float) -> None:
    task = _make(8, length=8, n_repetitions=2000, p_noise=p_noise, seed=7)
    steps = _pull(task, task.total_steps)
    rate = sum(lbl.meta["noised"] for _, lbl in steps) / len(steps)
    assert abs(rate - p_noise) < 0.02


def test_noise_substitutes_to_a_different_symbol() -> None:
    task = _make(5, length=5, n_repetitions=500, p_noise=0.6, seed=3)
    clean = _make(5, length=5, n_repetitions=500)
    clean_targets = [lbl.target for _, lbl in _pull(clean, clean.total_steps)]
    for i, (raw, lbl) in enumerate(_pull(task, task.total_steps)):
        if lbl.meta["noised"]:
            assert raw != lbl.meta["clean_symbol"]
            assert raw in task.categories
        assert lbl.target == clean_targets[i]  # labels never corrupted


def test_zero_noise_never_substitutes() -> None:
    task = _make(4, n_repetitions=1000, p_noise=0.0, seed=1)
    assert not any(lbl.meta["noised"] for _, lbl in _pull(task, task.total_steps))

# Acceptance 3: Seeding is reproducible
def test_same_seed_reproduces_stream()->None:
    a = _make(6, length=6, n_repetitions=200, p_noise=0.3, seed=42)
    b = _make(6, length=6, n_repetitions=200, p_noise=0.3, seed=42)
    sa = [(r, l.target, l.meta["noised"]) for r, l in _pull(a, a.total_steps)]
    sb = [(r, l.target, l.meta["noised"]) for r, l in _pull(b, b.total_steps)]
    assert sa == sb

def test_reset_rewinds_stream()->None:
    task=_make(6,n_repetitions=50,p_noise=0.4,seed=9)
    first=[r for r, _ in _pull(task,task.total_steps)]
    task.reset(seed=9)
    again=[r for r, _ in _pull(task,task.total_steps)]
    assert first == again

def test_different_seed_diverges()->None:
    a= _make(6,length=6,n_repetitions=200,p_noise=0.4,seed=9)
    b=_make(6, length=6, n_repetitions=200,p_noise=0.4,seed=2)
    va=[r for r, _ in _pull(a,a.total_steps)]
    vb=[r for r, _ in _pull(b,b.total_steps)]
    assert va!=vb

# --- Registry + build_task + CategoryEncoder pairing ------------------------

def test_registered_and_buildable() -> None:
    task = build_task(_cfg(4))  # build_task resets internally
    assert isinstance(task, SequenceTask)
    assert isinstance(task, Task)
    _raw, label = task.next_step()
    assert label.step == 0


def test_encoder_hint_is_category() -> None:
    assert _make(4).encoder_hint == "category"


def test_pairs_cleanly_with_category_encoder() -> None:
    task = _make(4, w=5, length=4, n_repetitions=2, p_noise=0.3, seed=5)
    enc = CategoryEncoder(n=task.n_input_dim, w=task.w, categories=task.categories)
    for raw, lbl in _pull(task, task.total_steps):
        assert enc.encode(raw).active_indices.size == task.w  # observed encodable
        enc.encode(lbl.target)                                # clean target encodable


# --- Validation -------------------------------------------------------------

def test_missing_alphabet_rejected() -> None:
    with pytest.raises(ValueError, match="alphabet"):
        SequenceTask(TaskConfig(type="sequence", n_input_dim=4, encoder_hint="category"))


def test_p_noise_out_of_range_rejected() -> None:
    with pytest.raises(ValueError, match="p_noise"):
        SequenceTask(_cfg(4, p_noise=1.5))


def test_noise_with_singleton_alphabet_rejected() -> None:
    with pytest.raises(ValueError, match="size >= 2"):
        SequenceTask(_cfg(1, p_noise=0.1))


def test_bad_length_rejected() -> None:
    with pytest.raises(ValueError, match="length"):
        SequenceTask(_cfg(4, length=0))


def test_duplicate_alphabet_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        SequenceTask(_cfg(["a", "a", "b"]))


def test_n_input_dim_not_multiple_of_k_rejected() -> None:
    cfg = TaskConfig(type="sequence", n_input_dim=10, encoder_hint="category",
                     params={"alphabet": 4})
    with pytest.raises(ValueError, match="multiple of the alphabet"):
        SequenceTask(cfg)


def test_bool_alphabet_rejected() -> None:
    with pytest.raises(TypeError):
        SequenceTask(_cfg(True))

