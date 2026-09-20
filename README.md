# AI-Driven Wireless Communication Network Optimization Using Machine Learning

Short title: **AI-Based Wireless Network Optimization**

This repository is a final-year B.Tech Electronics and Communication Engineering project scaffold for studying whether machine-learning-based wireless-network decisions can improve system performance under changing network conditions.

The project will be built incrementally. Phase 1 creates the repository architecture, configuration system, documentation placeholders, and sanity tests. Later phases will add NS-3 installation, C++ wireless simulations, dataset generation, ML training, optimization, evaluation, and research plots.

## Research Question

Can machine-learning-based decision-making improve wireless network performance under dynamically changing communication conditions compared with conventional optimization approaches?

## Phase 1 Scope

Completed in this phase:

- GitHub-ready repository layout.
- Configuration-driven project settings in `config/config.yaml`.
- Python package skeleton under `src/`.
- Baseline deterministic channel-selection policy.
- Metric formula utilities for throughput, PDR, BER, latency, energy, and spectrum efficiency.
- Documentation skeleton for architecture, methodology, and research gap.
- Placeholder directories for NS-3 simulations, datasets, results, and notebooks.
- Tests that can run with `pytest` after dependency installation.

Not completed yet:

- NS-3 installation.
- NS-3 C++ wireless simulation.
- Real generated dataset.
- Random Forest and XGBoost training runs.
- AI vs conventional numerical results.
- Literature-review citations.

All experimental numbers remain **Experiment pending** until real simulations are executed.

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

## Planned Development Phases

1. Repository and project architecture.
2. NS-3 installation and verification.
3. Basic wireless network simulation.
4. Dynamic wireless conditions.
5. Dataset generation.
6. Conventional baseline.
7. Python preprocessing.
8. Random Forest model.
9. XGBoost model.
10. Optimization integration.
11. AI vs conventional experiments.
12. Metrics and visualization.
13. Testing.
14. Docker and reproducibility.
15. Research documentation.
16. Final report and presentation material.

## Baseline Choice

The first deterministic baseline is lowest-interference channel selection with deterministic tie-breaking by higher SNR and then lower channel ID. This is simple, reproducible, and directly comparable to an ML channel-selection model in later phases.

## Reproducibility Rule

Do not report numerical wireless performance until NS-3 simulations and evaluation scripts have actually been executed. Use "Experiment pending" in reports until then.

