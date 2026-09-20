# Methodology

## Phase 1

Create the repository structure, configuration, initial modules, and sanity tests.

## Later Phases

1. Verify NS-3 installation on macOS.
2. Build a basic wireless C++ scenario with one access point and multiple user nodes.
3. Add dynamic SNR, interference, traffic load, and user-count conditions.
4. Export raw simulation traces to CSV.
5. Preprocess and validate data in Python.
6. Train Random Forest and XGBoost models for channel selection.
7. Compare ML-selected channels against the deterministic baseline using identical network conditions.
8. Generate plots and report-ready tables only after simulations have been executed.

## Experimental Rule

Numerical results must not be fabricated. Until a script has produced a metric from real simulation output, mark it as "Experiment pending".

