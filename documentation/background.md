# HALO — Background

> Heterarchical Associative Learning Orchestration

HALO is a Python research framework for biologically inspired cortical learning.
It is being developed as the implementation substrate of a PhD research project,
with the long-term goal of supporting **multi-context sparse consensus learning**
across a population of independent cortical units that communicate
heterarchically rather than through a strict layered hierarchy.

This document captures the *why* of the project: the scientific motivation, the
architectural decisions, the conventions, and the current state of the
implementation. Implementation specifics belong in inline docstrings; this file
is the conceptual map.

---

## 1. Motivation

Most current AI systems are built around backpropagation through deep,
strictly hierarchical networks. The cortex is neither. It is:

- **Sparse**: only ~2 % of neurons are active at any moment.
- **Local in its learning**: synaptic changes depend on pre- and post-synaptic
  activity, not on a global error gradient.
- **Heterarchical**: cortical regions interact peer-to-peer (lateral, apical,
  thalamic loops), not just feed-forward.
- **Predictive**: every region maintains a temporal model of its inputs and is
  driven primarily by surprise / prediction error.

HALO is an attempt to build a working, testable framework that respects all
four of these properties. It draws directly from:

- **HTM / Cortical Learning Algorithm** — Hawkins et al. 2011, Hawkins & Ahmad 2016
- **NAA (Neural Associations Algorithm)** — Dobric / NeoCortexAPI
- **Thalamocortical loops** — Sherman & Guillery 2006
- **TRN-like selective inhibition** — Crick 1984, Pinault 2004
- **Dopamine-modulated plasticity** — Schultz 1997

The framework is explicitly research-oriented. Correctness, extensibility, and
biological fidelity are prioritised over raw speed.

---

## 2. Theoretical Background

This section unpacks the three conceptual pillars on which HALO rests:
the thalamus as a relay-and-arbitration hub, *heterarchy* as an
alternative to strict hierarchies, and the family of voting mechanisms
that turn a population of independent cortical units into a coherent
output.

### 2.1 The thalamus and thalamocortical loops

In mammalian cortex, the thalamus is not merely a passive relay between
sensory organs and cortex. Sherman & Guillery (2006) distinguish two
classes of thalamic relay:

- **First-order relays** carry peripheral sensory signals to a primary
  cortical area (e.g. LGN → V1).
- **Higher-order relays** carry signals *between cortical areas* via the
  thalamus (e.g. pulvinar mediating V1 ↔ V2 communication). These loops
  are believed to coordinate attention, binding, and context switching
  across regions that have no direct cortico-cortical connection.

Two properties matter for HALO:

1. **Relays are gated and modulated.** The same thalamic neuron can pass,
   amplify, or suppress its input depending on cortical feedback and
   neuromodulators. The thalamus is not a simple wire — it is a
   reliability- and context-sensitive switchboard.
2. **The Thalamic Reticular Nucleus (TRN)** wraps the thalamus and
   provides selective inhibition. Crick's 1984 *searchlight hypothesis*
   and Pinault's 2004 review describe the TRN as the structure that
   decides *which* thalamic channels are heard at any given moment. It
   gates by ambiguity / entropy as much as by raw strength.

In HALO this maps onto two layers:

| Biology | HALO component | Role |
|---|---|---|
| Higher-order thalamic relay | `ThalamicLayer` | Aggregates per-unit SDRs into a broadcast signal, **weighted by each unit's current reliability**. |
| TRN selective inhibition | `TRNGatingLayer` | Suppresses per-unit SDRs whose **population entropy** exceeds a threshold — i.e. ambiguous units are silenced before they reach consensus. |

The thalamic layer is therefore not a softmax over outputs; it is a
trust-aware mixer that respects the same kind of *contextual gating* the
biological thalamus performs.

### 2.2 Heterarchy vs hierarchy

A *hierarchy* is a strict acyclic stack: information flows up (or down)
through ordered levels, and each level only talks to its immediate
neighbours. Deep neural networks are the canonical example.

A *heterarchy* (Cumming 1969; McCulloch 1945, *"A Heterarchy of Values
Determined by the Topology of Nervous Nets"*) replaces the partial order
with a *graph*: nodes at the same conceptual level can talk to each
other, and a node's influence depends on context rather than position.
The cortex is heterarchical in several senses:

- Cortical regions at the same processing depth project laterally to
  each other.
- Apical dendrites of layer-5 neurons receive feedback / contextual
  input from "higher" areas, but this input *modulates* rather than
  *drives* the cell (Larkum 2013; Hawkins & Ahmad 2016).
- Multiple parallel pathways (dorsal/ventral, sensory/attentional) can
  arbitrate without one being strictly upstream of the other.

HALO realises heterarchy through:

- **A population of independent `CorticalUnit` instances** that each
  receive the same raw input but learn distinct sparse encodings.
- **`HeterarchicalLayer`**, which maintains learned lateral weight
  matrices `W[from → to]` between every pair of units. The lateral
  contribution is added to a unit's pre-inhibition overlap scores *only
  for columns that already have feedforward support* — directly
  honouring the modulate-not-drive principle from Hawkins & Ahmad 2016.
- **No single "top" unit.** Consensus emerges from the population
  rather than being read out from a designated apex.

### 2.3 The voting / consensus problem

Once N independent units have each produced an SDR, the system has to
collapse those N candidates into a single output. This is fundamentally
an **ensemble arbitration problem**, with three specifically biological
constraints that rule out most ML defaults:

1. The output must itself be a valid sparse distributed representation
   (same length, same sparsity), not a softmax probability vector.
2. The arbitration must respect *which units are currently trustworthy*
   — a unit that has been bursting (high prediction error) for the last
   few steps should contribute less.
3. The arbitration must respect *which units are currently confident* —
   a high-entropy SDR is uninformative and should be muted regardless of
   the unit's long-term reliability.

The literature offers several families of solutions. HALO categorises
them as follows.

#### 2.3.1 Pure union (bitwise OR)

Take the OR of every unit's SDR. Implemented as the `or` aggregation in
`ThalamicLayer`. Useful as a baseline — it preserves every active bit
but inflates sparsity linearly with N and ignores both reliability and
entropy. Not used in the consensus path.

#### 2.3.2 Reliability-weighted accumulation + top-k thresholding

Per-bit score `s_j = Σ_u w_u · sdr_u[j]`, where `w_u` is unit `u`'s
trust score, followed by selecting the top-k bits to enforce the target
output sparsity. This is the `weighted_sum` aggregation in
`ThalamicLayer` and also the basis of `ConsensusEngine.aggregate()`.
Properties:

- Reduces to majority vote when all `w_u` are equal.
- Suppresses low-trust units gracefully without removing them.
- Ties are broken deterministically (`np.lexsort` on descending score,
  ascending index) so experiments are reproducible.
- All-zero weight vector returns an empty SDR — a unit population
  with zero collective trust should not be forced to produce activity.

#### 2.3.3 Entropy-based gating (TRN-style)

For each unit's SDR, compute the active-bit population entropy `H`. If
`H` exceeds an `entropy_threshold` the SDR is replaced with the empty
SDR before it reaches consensus. This is what `TRNGatingLayer`
implements. Compared with reliability weighting:

- Reliability is a *slow* signal: it accumulates across many steps via
  the dopamine update.
- Entropy gating is a *fast* signal: it applies per-step, regardless of
  past performance.

The two are complementary — slow trust × fast confidence — and both feed
the consensus engine in the current pipeline.

#### 2.3.4 Proposed voting mechanisms (research roadmap)

The current consensus layer is deliberately a minimal weighted vote
(method `"weighted_vote"` in `ConsensusConfig`). Several richer
mechanisms are on the roadmap for Phase 2:

- **Borda / rank-aggregated voting.** Instead of treating every active
  bit as binary, rank each unit's bits by overlap strength and combine
  ranks across units. Less sensitive to per-unit miscalibration of
  thresholds.
- **Per-context committees.** Maintain *multiple* consensus heads, each
  trained to specialise on a context signature; let a gating SDR select
  which committee speaks. This is the natural home of *multi-context
  sparse consensus*, the project's Phase-2 direction.
- **Approval voting with entropy-weighted thresholds.** Each unit
  "approves" a bit when its activation exceeds a confidence-scaled
  threshold; the consensus retains bits approved by at least `m` units.
  Provides graceful degradation when some units are silenced.
- **Predictive-coding residual voting.** Each unit votes not on its raw
  SDR but on the *residual* between its prediction and the actual
  feedforward input. Units that contribute novel error get more weight.
  Aligns the dopamine signal (already prediction-accuracy-based) with
  the consensus mechanism.
- **Heterarchical attention.** Use the learned lateral weights from
  `HeterarchicalLayer` not only for biasing encoders but also as a
  similarity / attention kernel during voting — letting strongly
  connected units form coalitions.

These mechanisms are not yet implemented. They are listed here so the
design rationale is captured for future work, and so the
`ConsensusEngine` interface can be extended without breaking changes.

---

## 3. Hard Design Constraints

These are non-negotiable rules of the codebase. Every contribution must respect
them.

1. **Python 3.10+ only.** Built-in generics (`list[int]`, `dict[str, T]`), `|`
   union types.
2. **No backpropagation.** Learning is strictly local — Hebbian, STDP-style,
   or HTM/NAA permanence rules.
3. **No deep-learning frameworks.** PyTorch, TensorFlow, JAX are off-limits
   unless explicitly justified for a single experiment.
4. **No hidden magic numbers.** Every parameter lives in a YAML config and
   passes through a validated dataclass in `halo/config/schema.py`.
5. **ABCs everywhere.** Each major component is an abstract base class in
   `halo/core/`; concrete implementations are swappable via configuration.
6. **SDR is the wire format.** All inter-component signals are
   `halo.core.sdr.SDR` instances — never raw boolean arrays.
7. **Reliability is owned by `ReliabilityModule`.** No other component may
   mutate trust scores.
8. **Logging, not `print`.** Each module gets its own logger via
   `logging.getLogger(__name__)`.

---

## 4. Architecture

### 4.1 Data flow

```
Input
  │
  ▼
┌────────────────────────────────────────────────────────────────────┐
│  Cortical Units (1..N, independent HTM/NAA modules)                │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐   │
│  │  SpatialPooler  │   │  SpatialPooler  │   │  SpatialPooler  │   │
│  │       +         │   │       +         │   │       +         │   │
│  │ TemporalMemory  │   │ TemporalMemory  │   │ TemporalMemory  │   │
│  └─────────────────┘   └─────────────────┘   └─────────────────┘   │
│         │      ▲                │      ▲                │      ▲   │
│         │      └────────────────┼──────┴────────────────┘      │   │
│         │      HeterarchicalLayer (learned lateral biases)     │   │
│         ▼                                                          │
└────────────────────────────────────────────────────────────────────┘
          │                       │                       │
          ▼                       ▼                       ▼
┌────────────────────┐   ┌────────────────────┐   ┌────────────────────┐
│ ThalamicLayer      │   │  TRN Gating        │   │  ConsensusEngine   │
│ reliability-       │   │  entropy-based     │   │  weighted vote     │
│ weighted aggreg.   │   │  inhibition        │   │  over gated SDRs   │
└────────────────────┘   └────────────────────┘   └────────────────────┘
                                                            │
                                                            ▼
                                          ReliabilityModule  ◄── per-unit
                                          (dopamine update)      prediction
                                                            │    accuracy
                                                            ▼
                                                       Final SDR
```

### 4.2 Component responsibilities

| Component | Role |
|---|---|
| `CorticalUnit` | HTM SP + TM. Encodes raw input into a column-level SDR via `encode()`, runs Temporal Memory via `temporal_step()`, updates permanences and grows segments via `learn()`. Exposes `prediction_accuracy` for the dopamine signal. |
| `HeterarchicalLayer` | Learned lateral connections between peer cortical units. Apical-style modulation (modulates, never drives). Hebbian weight update. CSR-sparse storage. |
| `ThalamicLayer` | Reliability-weighted aggregation of per-unit column SDRs into a broadcast signal. Supports `or` (baseline) and `weighted_sum` (top-k thresholded). |
| `TRNGatingLayer` | Entropy-based selective inhibition: suppresses per-unit SDRs whose population entropy exceeds a threshold. |
| `ReliabilityModule` | Per-unit trust score in [min_score, max_score]. Updated via dopamine signal in [-1, 1]. Sole writer of trust scores. |
| `ConsensusEngine` | Weighted voting over gated SDRs (weighted by reliability) → unified consensus SDR. |
| `HALOPipeline` | Wires all components together. One `step()` runs the full loop end-to-end. |

### 4.3 Repository layout

```
halo/
  core/          # Abstract base classes, SDR dataclass, shared data structures
  models/        # CorticalUnit (HTM SP + TM)
  layers/        # heterarchical.py, thalamic.py, trn.py
  reliability/   # Trust scoring and dopamine-like update rules
  consensus/     # Voting and aggregation strategies
  orchestration/ # HALOPipeline — wires components together
  config/        # YAML schema, dataclass validation, loader
  encoders/      # ScalarEncoder, CategoryEncoder
  utils/         # Logging, metrics
experiments/     # Standalone experiment scripts (one file per experiment)
configs/         # YAML config files for experiments
tests/           # Unit + integration tests, mirroring halo/ structure
documentation/   # Project-wide design docs (this folder)
```

---

## 5. Conventions

### SDR

Sparse Distributed Representations are the universal currency. An `SDR`
carries a fixed-length boolean array plus metadata (originating unit ID and
timestamp). The framework standardises on ~2 % sparsity by default.

### Configuration

All experiment parameters are validated dataclasses in `halo/config/schema.py`
and loaded from YAML. Pipeline code never reads scalar literals — it reads
config attributes. The `__post_init__` of each config dataclass enforces
ranges, ratios, and compatibility constraints (e.g. `min_score ≤ initial_score
≤ max_score`).

### Reinforcement signal

The dopamine-like signal is a scalar in `[-1.0, 1.0]`. Reliability scores per
unit are in `[0.0, 1.0]`. The current pipeline derives the per-unit signal as

```
dopamine_u = unit.prediction_accuracy * 2 - 1
```

where `prediction_accuracy` is the fraction of last step's active columns that
were correctly predicted by the unit's Temporal Memory. Units with strong
temporal models receive positive reinforcement; bursting units receive
negative reinforcement. Each unit gets its own signal — reliability scores
diverge naturally over time.

### Testing

`pytest` is the only test runner. Tests mirror the source layout
(`tests/layers/test_thalamic.py`, `tests/models/test_cortical_unit.py`, …).
Integration tests for the full pipeline live under `tests/orchestration/`.
Tests are expected to be deterministic — RNGs are seeded explicitly.

### Commands

```bash
pip install -e ".[dev]"           # editable install with dev extras
pytest                            # run the full suite
pytest tests/path/to/file.py      # one file
pytest tests/path/to/file.py::test_name   # one test
ruff check halo/ tests/           # lint
mypy halo/                        # type-check (strict)
```

---

## 6. Implementation Status

### Phase 1 — Baseline Stabilization (current)

| Issue | Title | Status |
|---|---|---|
| #14 | HTM Spatial Pooler | ✅ merged |
| #16 | HTM Temporal Memory | ✅ merged |
| #17 / #12 | ScalarEncoder + CategoryEncoder | ✅ merged |
| #18 | HeterarchicalLayer apical-style lateral connections | ✅ merged |
| #21 | TM sequence integration tests + 2 TM bug fixes | ✅ merged |
| #10 | ThalamicLayer reliability-weighted aggregation | ✅ merged |
| #11 | Pipeline per-unit prediction-accuracy dopamine signal | ✅ merged |
| #25 | HeterarchicalLayer CSR-sparse storage | 🟡 in PR |

### Notable engineering wins so far

- **TM correctness fix**: `temporal_step()` originally snapshotted the
  previous active/winner cells at the *end* of the method, meaning Phase 1
  always saw state from two timesteps ago. Now snapshotted at the top.
- **TM learn guard**: `learn()` had a single early-return on `_last_input
  is None` that silently disabled TM segment adaptation. Now SP and TM paths
  are guarded independently.
- **Pipeline TM regression**: `HALOPipeline.step()` never called
  `temporal_step()` until issue #11 — TM cells were always empty and the
  dopamine signal was meaningless. The new pipeline calls TM explicitly
  between `encode()` and `learn()`.
- **Thalamic reliability weighting**: `weighted_sum` aggregation now does
  reliability-weighted accumulation followed by top-k thresholding with
  deterministic tie-breaking (`np.lexsort`). Missing trust scores default to
  `0.0` (suppression), not `1.0`. All-zero weights produce an empty SDR.
- **HeterarchicalLayer memory**: dense `(n_columns, n_columns)` storage
  replaced with a CSR triple (`indptr`, `indices`, `data`) plus cached
  `row_idx`. At baseline config (n=2048, 4 units all-to-all) memory dropped
  from ~252 MB to ~12 MB (21× reduction). `compute_biases()` now uses
  `np.bincount` over the cached row indices instead of dense matmul.

---

## 7. Next Phase

Once Phase 1 is fully merged, the project moves into **Phase 2 — Multi-context
Sparse Consensus Learning**, the actual PhD research direction (referred to
internally as direction *D*). Expected work:

- A real input dataset beyond random noise (likely streaming sensor data
  and/or a small text/sequence benchmark).
- An evaluation harness for measuring sequence-prediction accuracy and
  consensus stability across contexts.
- Experiments where the population of cortical units is exposed to multiple
  overlapping contexts simultaneously and the consensus layer must
  arbitrate between competing predictions.
- Comparisons against single-unit HTM baselines to demonstrate that the
  heterarchical population produces measurable benefits.

---

## 8. References

- Hawkins, J. et al. (2011). *Hierarchical Temporal Memory — Cortical Learning
  Algorithm white paper*. Numenta.
- Hawkins, J. & Ahmad, S. (2016). *Why Neurons Have Thousands of Synapses, a
  Theory of Sequence Memory in Neocortex*. Frontiers in Neural Circuits.
- Larkum, M. (2013). *A cellular mechanism for cortical associations: an
  organizing principle for the cerebral cortex*. Trends in Neurosciences.
- McCulloch, W. S. (1945). *A Heterarchy of Values Determined by the Topology
  of Nervous Nets*. Bulletin of Mathematical Biophysics.
- Sherman, S. M. & Guillery, R. W. (2006). *Exploring the Thalamus and Its
  Role in Cortical Function*. MIT Press.
- Crick, F. (1984). *Function of the thalamic reticular complex: the
  searchlight hypothesis*. PNAS.
- Pinault, D. (2004). *The thalamic reticular nucleus: structure, function
  and concept*. Brain Research Reviews.
- Schultz, W., Dayan, P. & Montague, P. R. (1997). *A neural substrate of
  prediction and reward*. Science.
- Dobric, D. (NeoCortexAPI). *SpatialPooler.cs, TemporalMemory.cs,
  NeuralAssociationsAlgorithm.cs*. GitHub.
