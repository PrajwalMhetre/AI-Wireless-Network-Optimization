# AURORA-RIS

AURORA is a deterministic, simulator-agnostic optimization core. It builds
CSI-derived graphs, supports a torch-only GNN+GRU policy (without
`torch-geometric`), quantizes RIS phases, scores uncertainty/OOD, gates risky
actions, and uses coordinate refinement before evaluating a proposal.

The default simulator is a NumPy link-budget fallback so CI and development do
not require NVIDIA Sionna. A Sionna adapter can consume the same
`Scenario`/`SimulationResult` interfaces when Sionna is available; no measured
performance is implied by the fallback.

## Run

```bash
uvicorn backend.app.main:app --reload
python experiments/run_aurora.py
```

All random generation accepts an explicit seed. `/health` identifies the
simulator and whether torch is installed. `/optimize` returns channels, phases,
gate status, and metrics.
