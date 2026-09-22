# AURORA-RIS

Adaptive Uncertainty-Aware Self-Correcting RIS Optimization for dynamic 6G
wireless networks.

This repository contains a reproducible, research-oriented NumPy fallback
simulation and an optional PyTorch GNN + GRU policy. The AURORA controller
estimates model confidence and distribution shift, accepts safe predictions,
or invokes a classical RIS phase refinement when the confidence gate rejects
them. Difficult observations are retained in an active-learning buffer.

The implementation is designed for studying whether uncertainty-aware
machine-learning decisions can improve wireless-network performance under
changing network conditions. It does not claim measured hardware or
unexecuted experimental results.

The production path is under `backend/aurora`; the legacy `src/` package is
retained for compatibility with the original channel-selection scaffold.

## Research Question

Can machine-learning-based decision-making improve wireless network performance under dynamically changing communication conditions compared with conventional optimization approaches?

## Implemented capabilities

- Configurable dynamic wireless scenarios with SNR, CSI error, mobility,
  environment, RIS size, and phase resolution.
- CSI feature extraction, k-nearest-neighbor wireless graphs, and a
  torch-only GNN + GRU forward path.
- Physically valid RIS phase quantization, confidence/uncertainty scoring,
  Mahalanobis OOD detection, and configurable confidence gating.
- Classical coordinate refinement, active-learning buffering, baselines,
  reproducible experiment execution, and FastAPI endpoints.
- Minimal React + TypeScript + Plotly dashboard and Docker packaging.

All performance figures produced by experiments are measurements from the
selected simulator. Unexecuted experiments remain **not yet measured**.

## AURORA-RIS backend

The reproducible AURORA-RIS core is under `backend/aurora`, with a FastAPI
surface in `backend/app`. It defaults to a deterministic NumPy simulator and
does not require Sionna or `torch-geometric`; torch is used by the optional
GNN+GRU policy when installed. Start the API with:

```bash
uvicorn backend.app.main:app --reload
python experiments/run_aurora.py
```

The `/optimize` response includes quantized RIS phases, a confidence/OOD gate,
classical coordinate refinement, model/active-learning metadata, and auditable
throughput/SINR/energy metrics. The same validated request is available at
`/api/simulation/run`, `/api/optimization/run`, `/api/aurora/predict`, and
`/api/aurora/optimize`; discovery endpoints are provided under
`/api/experiments`, `/api/metrics`, `/api/models`, and
`/api/active-learning/retrain`. Requests can set `snr_db`, `csi_error`,
`mobility`, `environment`, and `phase_resolution`.
The fallback is for development and reproducibility, not measured hardware
performance.

## Repository Structure

```text
.
├── config/
├── data/
│   ├── raw/
│   ├── processed/
│   └── datasets/
├── docs/
├── experiments/
├── notebooks/
├── results/
│   ├── figures/
│   ├── metrics/
│   └── reports/
├── simulation/
│   └── ns3/
│       ├── scenarios/
│       ├── models/
│       └── scripts/
├── src/
└── tests/
```

## Setup

```bash
cd AI-Wireless-Network-Optimization
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run Tests

After installing dependencies:

```bash
python -m pytest
```

Expected output for Phase 1:

```text
all tests pass
```

If dependencies are not installed yet, a standard-library smoke test can still be run:

```bash
python3 -m unittest discover -s tests
```

## API endpoints

The FastAPI application exposes `/health`, `/optimize`, and the specified
`/api/simulation/run`, `/api/optimization/run`, `/api/aurora/predict`,
`/api/aurora/optimize`, `/api/experiments`, `/api/experiments/run`,
`/api/experiments/{id}`, `/api/metrics`, `/api/models`, and
`/api/active-learning/retrain` routes.

## Baseline Choice

The first deterministic baseline is lowest-interference channel selection with deterministic tie-breaking by higher SNR and then lower channel ID. This is simple, reproducible, and directly comparable to an ML channel-selection model in later phases.

## Reproducibility Rule

Do not report numerical wireless performance until NS-3 simulations and evaluation scripts have actually been executed. Use "Experiment pending" in reports until then.
