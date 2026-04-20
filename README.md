# HALO — Heterarchical Associative Learning Orchestration

A biologically inspired cortical learning framework implementing heterarchical
associative processing without backpropagation.

## Architecture

- **CorticalUnit** — Sparse Distributed Representation encoder with Hebbian learning
- **HeterarchicalLayer** — Lateral signal mixing across peer units (no strict hierarchy)
- **ThalamicLayer** — Relay and aggregation of cortical outputs (Sherman & Guillery 2006)
- **TRNGatingLayer** — Entropy-driven selective inhibition (Crick 1984; Pinault 2004)
- **ReliabilityModule** — Dopamine-modulated trust scores (Schultz 1997)
- **ConsensusEngine** — Weighted-vote SDR consensus across units
- **HALOPipeline** — End-to-end orchestration

## Quickstart

```bash
pip install -e ".[dev]"
python experiments/run_baseline.py
pytest tests/
```

## Design Principles

- No backpropagation; Hebbian-style local learning rules only
- All inter-layer communication via typed `SDR` objects
- Configuration via pure dataclasses loaded from YAML
- `ReliabilityModule` is the sole owner of trust-score mutations
