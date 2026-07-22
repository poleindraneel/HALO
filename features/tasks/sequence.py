"""SequenceTask (issue #33, Phase-2): cyclic A -> B -> C -> ... symbol streams.

First structured task (counterpart to RandomNoiseTask). Emits a deterministic
stream of category symbols; each step's TaskLabel carries the CLEAN next symbol
as `target`. encoder_hint="category" so the harness pairs it with CategoryEncoder.
"""
from __future__ import annotations
import string
from typing import Any
import numpy as np
from features.tasks.base import EncoderHint, Task, TaskInput, TaskLabel
from features.tasks.config import TaskConfig
from features.tasks.registry import register_task

__all__= ["SequenceTask"]

def _build_alphabet(spec: int|list[str])->list[str]:
    """ Resolve an alphabet spec into a list of unique symbol strings"""
    if isinstance(spec,bool): #bool is an int subclass: reject explicitely
        raise TypeError("params['alphabet'] must be an int size or a list of strings")
    if isinstance(spec,int):
        if spec<1:
            raise ValueError(f'alphabet size K must be >=1, got {spec}')
        if spec<=26:
            return list(string.ascii_uppercase[:spec])
        return [f"S{i}" for i in range(spec)]
    if isinstance(spec,(list,tuple)):
        symbols=list(spec)
        if not symbols:
            raise ValueError("Alphabet list must be non-empty")
        if not all(isinstance(s,str) for s in symbols):
            raise TypeError("Explicit alphabet entries must be strings")
        if len(symbols)!=len(set(symbols)):
            raise ValueError("Alphabet must not contain duplicates")
        return symbols
    raise TypeError("params['alphabets'] must be an int size or a list of strings")
@register_task
class SequenceTask(Task):
    """ Deterministic cyclic symbol-sequence task with optional substitution noise. """
    name="sequence"
    def __init__(self, config: TaskConfig):
        super().__init__(config)
        p=config.params


        if "alphabet" not in p:
            raise ValueError("SequenceTask requires params['alphabet']")
        self._alphabet:list[str]=_build_alphabet(p["alphabet"])
        k=len(self._alphabet)

        self._length:int=int(p.get("length",k))
        if self._length<1:
            raise ValueError(f"params['length'] must be >=1, got {self._length}")
        self._n_repetitions:int=int(p.get("n_repetitions",1))
        if self._n_repetitions <1:
            raise ValueError(f"params['n_repetitions'] must be >=1, got {self._n_repetitions}")
        
        self._p_noise : float=float(p.get("p_noise",0.0))
        if not 0.0<=self._p_noise <=1.0:
            raise ValueError(f"params['p_noise'] must be in [0,1], got {self._p_noise}")
        if self._p_noise>0.0 and k < 2:
            raise ValueError("p_noise > 0 requires an alphabet of size >= 2")
        #n_input_dim must size a CategoryEncoder exactly: n=K*w,w>=1.
        self._n_input_dim: int = config.n_input_dim
        if self._n_input_dim % k!=0:
            raise ValueError(
                f"n_input_dim({self._n_input_dim}) must be a multiple of the "
                f"alphabet size K={k} (so w = n_input_dim //k is an integer)"
            )
        self._w:int =self._n_input_dim//k
        if self._w<1:
            raise ValueError(
                f"n_input_dim({self._n_input_dim}) too small for k={k}"
                f"needs n_input_dim>=K so w>=1"
            )
        

        #Clean base sequence, cycling the alphabet, then repeated.
        #Targets = clean *next* symbol, cyclic at the wrap.
        base=[self._alphabet[i%k]for i in range(self._length)]
        self._clean_stream:list[str] = base * self._n_repetitions
        total=len(self._clean_stream)
        self._targets:list[str]=[
            self._clean_stream[(t+1)%total] for t in range(total)
        ]
        self._idx_of:dict[str,int]={s:i for i,s in enumerate(self._alphabet)}
        self._step=0
        self._rng:np.random.Generator|None=None # set in reset().

    def reset(self,seed:int)->None:
        """Reseed and rewind: same seed reproduces the exact stream"""
        self._rng=np.random.default_rng(seed)
        self._step=0
    def next_step(self)->tuple[TaskInput,TaskLabel]:
        if self._rng is None:
            raise RuntimeError("next_step() called before reset()")
        
        total=len(self._clean_stream)
        idx=self._step %total #cycle safely past the designed stream length
        clean=self._clean_stream[idx]

        noised=self._p_noise>0.0 and bool(self._rng.random()<self._p_noise)
        if noised:
            k=len(self._alphabet)
            offset=int(self._rng.integers(1,k)) #1 . .k-1 -> guaranteed different.
            observed=self._alphabet[(self._idx_of[clean]+offset)%k]
        else:
            observed=clean

        label=TaskLabel(
            step=self._step,
            target=self._targets[idx],
            context_id=None, # Single - Context task
            meta={"noised":noised,"clean_symbol":clean},
        )
        self._step+=1
        return observed, label
    @property
    def n_input_dim(self)->int:
        return self._n_input_dim
    
    @property
    def encoder_hint(self) -> EncoderHint:
        return "category"
    
    #-------------------Introspection (used by the harness/tests)---------------------------------------------
    @property
    def categories(self)->list[str]:
        """Full ordered symbol vocabulary(for building the CategoryEncoder)."""
        return list(self._alphabet)
    
    #Alias kept for readability at call sites
    alphabet=categories

    @property
    def w(self)->int:
        """Active bits per category (n_input_dim//k)."""
        return self._w
    
    
    @property
    def length(self)->int:
        return self._length
    
    @property
    def n_repetitions(self)->int:
        return self._n_repetitions
    
    @property
    def p_noise(self)->float:
        return self._p_noise
    
    @property
    def total_steps(self)->int:
        """Length of one full designed stream(length*n_repetitions)."""
        return len(self._clean_stream)
    @property
    def clean_stream(self)->list[str]:
        return list(self._clean_stream)
    
    @property
    def targets(self)->list[str]:
        return list(self._targets)