# HALO — Heterarchical Associative Learning Orchestration

## Project Overview

HALO is a Python research framework for biologically inspired cortical learning. It implements Sparse Distributed Representations (SDRs) and Cortical Learning Algorithms (HTM/CLA/NAA principles) with heterarchical interaction between independent cortical units.

**Python 3.10+. No backpropagation. No deep learning frameworks (PyTorch, TensorFlow) unless strictly justified.**

---

## Architecture

The data flow pipeline is:

```
Input → Cortical Units → Heterarchical Communication → Thalamic Integration → TRN Gating → Consensus Engine → Output
```

### Core Layers

| Component | Role |
|---|---|
| `CorticalUnit` | HTM/NAA-inspired module; learns SDRs from an input stream independently |
| `HeterarchicalLayer` | Lateral + feedback communication between units (no strict hierarchy) |
| `ThalamicLayer` | Aggregates and routes SDR outputs from multiple cortical units |
| `TRNGatingLayer` | Entropy-based conflict detection; suppresses low-confidence signals |
| `ReliabilityModule` | Per-unit trust score; updated via a dopamine-like reinforcement signal |
| `ConsensusEngine` | Weighted voting over cortical unit outputs → unified representation |

### Separation of Concerns

```
halo/
  core/          # Abstract base classes, SDR dataclass, shared data structures
  models/        # Cortical unit implementations (HTM variants, NAA, etc.)
  layers/        # Thalamic, TRN gating, heterarchical communication
  reliability/   # Trust scoring and dopamine-like update rules
  consensus/     # Voting and aggregation strategies
  orchestration/ # Pipeline wiring, experiment runner, unit coordination
  config/        # YAML schema definitions and config loader
  utils/         # Logging, metrics, serialization helpers
experiments/     # Standalone experiment scripts (one file per experiment)
configs/         # YAML config files for experiments
tests/           # Unit and integration tests mirroring halo/ structure
```

---

## Key Conventions

### Abstract Base Classes

All major components are defined as ABCs in `halo/core/`. Concrete implementations live in their respective subpackages and inherit from these interfaces. Never instantiate core ABCs directly.

```python
# halo/core/base.py pattern
from abc import ABC, abstractmethod

class CorticalUnitBase(ABC):
    @abstractmethod
    def encode(self, input_data: np.ndarray) -> SDR: ...

    @abstractmethod
    def learn(self, sdr: SDR) -> None: ...
```

### Type Hints

All public functions and methods must be fully annotated using `typing` / built-in generics (Python 3.10+ style: `list[int]` not `List[int]`). No `Any` without a comment justifying it.

### SDR Convention

SDRs are represented as `numpy` boolean arrays of fixed length. A dedicated `SDR` dataclass (in `halo/core/sdr.py`) wraps the array and carries metadata (sparsity, origin unit ID, timestamp). Always pass `SDR` objects between layers — never raw arrays.

### Configuration

All experiment-level parameters are loaded from YAML via `halo/config/`. No magic numbers in source code — reference config values. Use `dataclasses` or Pydantic to validate config schemas at load time.

```python
# Correct
config = load_config("configs/experiment_01.yaml")
unit = CorticalUnit(config.cortical)

# Wrong — hardcoded values
unit = CorticalUnit(n_columns=2048, sparsity=0.02)
```

### Logging

Use the standard `logging` module. Each module gets its own logger via `logging.getLogger(__name__)`. Do not use `print()` for diagnostic output in framework code. Log levels: `DEBUG` for step-by-step trace, `INFO` for pipeline events, `WARNING` for ambiguity/gating triggers.

### Reinforcement / Reliability Updates

The dopamine-like signal is a scalar in `[-1.0, 1.0]`. Reliability scores per unit are in `[0.0, 1.0]`. The `ReliabilityModule` owns all mutation of these scores — no other component may write to them directly.

### No Backpropagation

Learning rules must be local (Hebbian, STDP-inspired, or NAA-style). Any change introducing gradient-based optimization must be explicitly justified in the commit message.

---

## Commands

```bash
# Install (editable, with dev extras)
pip install -e ".[dev]"

# Run all tests
pytest

# Run a single test file
pytest tests/core/test_sdr.py

# Run a single test by name
pytest tests/core/test_sdr.py::test_sparsity_invariant

# Lint
ruff check halo/ tests/

# Type check
mypy halo/
```

---

## Experiment Structure

Each experiment is a standalone script in `experiments/` that:
1. Loads a YAML config from `configs/`
2. Wires components via `HALOPipeline`
3. Runs steps until convergence or a step limit
4. Logs results and optionally serializes final state

```python
# experiments/run_baseline.py
from halo.orchestration import HALOPipeline
from halo.config import load_config

config = load_config("configs/baseline.yaml")
pipeline = HALOPipeline(config)
pipeline.run()
```

---

## Design Constraints

- Cortical units must be **swappable** — wire via config, not hard imports.
- Gating strategies and reliability update rules are **pluggable** — use the ABC interfaces.
- Pipeline wiring belongs in `halo/orchestration/`, not in experiment scripts.
- Docstrings on biologically inspired mechanisms should cite the biological source (e.g., `# TRN-like inhibition: Crick 1984`).