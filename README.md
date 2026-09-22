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

## Output page

The dashboard provides a separate interactive output view. It sends the
scenario controls to `/api/aurora/optimize` and displays throughput, SINR,
confidence, OOD score, selected channels, optimizer invocation, and a
per-user Plotly chart.

```bash
# Terminal 1: backend
source .venv/bin/activate
uvicorn backend.app.main:app --reload --port 8000

# Terminal 2: dashboard
cd dashboard
npm install
npm run dev
```

Open `http://localhost:5173/#output`. The API proxy in
`dashboard/vite.config.ts` forwards requests to port 8000.

## How the project works

1. The scenario generator creates users, channel gains, interference,
   mobility, SNR, CSI error, environment, and RIS settings using a fixed seed.
2. CSI is normalized and converted into real node features. A k-nearest
   neighbor graph represents spatial relationships between users.
3. When PyTorch is installed, the GNN encodes spatial features and the GRU
   models temporal state. The controller predicts channels and RIS phases.
   Without PyTorch, the deterministic baseline remains available.
4. Policy confidence and predictive variance are combined with a Mahalanobis
   distance OOD score. The configurable gate accepts safe AI output or calls
   coordinate-descent classical phase refinement.
5. The simulator calculates per-user throughput, sum throughput, SINR, energy,
   and spectral efficiency. Low-confidence/OOD cases enter the active-learning
   buffer.
6. The output page presents the actual response returned by the API; it does
   not fabricate benchmark results.

## Deployment

For local deployment with API, dashboard, and PostgreSQL:

```bash
docker compose up --build
```

Then open `http://localhost:5173` for the output page and
`http://localhost:8000/docs` for API documentation. Stop services with
`docker compose down`; add `-v` only when you intentionally want to remove
the PostgreSQL volume.

For a server deployment, place a reverse proxy such as Nginx or Caddy in
front of ports 5173 and 8000, enable HTTPS, and set production database
credentials through environment variables rather than committing them.

## Baseline Choice

The first deterministic baseline is lowest-interference channel selection with deterministic tie-breaking by higher SNR and then lower channel ID. This is simple, reproducible, and directly comparable to an ML channel-selection model in later phases.

## Reproducibility Rule

Do not report numerical wireless performance until NS-3 simulations and evaluation scripts have actually been executed. Use "Experiment pending" in reports until then.
