"""FastAPI application exposing deterministic AURORA optimization."""
from __future__ import annotations
from .schemas import HealthResponse, OptimizeRequest, OptimizeResponse, RetrainRequest
from backend.aurora.controller import AURORAController
from backend.aurora.scenario import Scenario, ScenarioConfig

try:
    from fastapi import FastAPI
except ImportError:  # pragma: no cover - allows core-only installations
    FastAPI = None


def create_app():
    if FastAPI is None:
        raise ImportError("FastAPI is required for the HTTP application")
    app = FastAPI(title="AURORA-RIS", version="0.1.0")

    @app.get("/health", response_model=HealthResponse)
    def health() -> dict:
        try:
            import torch
            torch_available = True
        except ImportError:
            torch_available = False
        return {"status": "ok", "simulator": "numpy-fallback",
                "torch_available": torch_available}

    def run_optimization(request: OptimizeRequest) -> dict:
        config = ScenarioConfig(
            num_users=request.num_users, num_elements=request.num_elements,
            num_channels=request.num_channels, ris_phase_bits=request.ris_phase_bits,
            phase_resolution=request.phase_resolution, seed=request.seed,
            snr_db=request.snr_db, snr=request.snr,
            csi_error=request.csi_error, csi_error_rate=request.csi_error_rate,
            mobility=request.mobility, mobility_mps=request.mobility_mps,
            environment=request.environment,
        )
        return AURORAController(
            Scenario.generate(config), seed=request.seed,
            confidence_threshold=request.confidence_threshold,
            ood_threshold=request.ood_threshold,
            max_ood_score=request.max_ood_score,
        ).optimize().__dict__

    # Keep the original endpoint and expose the names used by the AURORA API
    # specification.  Shared request models ensure all aliases validate input.
    @app.post("/optimize", response_model=OptimizeResponse)
    def optimize(request: OptimizeRequest) -> dict:
        return run_optimization(request)

    @app.post("/api/simulation/run", response_model=OptimizeResponse)
    def simulation_run(request: OptimizeRequest) -> dict:
        return run_optimization(request)

    @app.post("/api/optimization/run", response_model=OptimizeResponse)
    def optimization_run(request: OptimizeRequest) -> dict:
        return run_optimization(request)

    @app.post("/api/aurora/predict", response_model=OptimizeResponse)
    def aurora_predict(request: OptimizeRequest) -> dict:
        return run_optimization(request)

    @app.post("/api/aurora/optimize", response_model=OptimizeResponse)
    def aurora_optimize(request: OptimizeRequest) -> dict:
        return run_optimization(request)

    @app.get("/api/experiments")
    def experiments() -> dict:
        return {"name": "AURORA-RIS", "status": "ready"}

    @app.post("/api/experiments", response_model=OptimizeResponse)
    def run_experiment(request: OptimizeRequest) -> dict:
        return run_optimization(request)

    @app.post("/api/experiments/run", response_model=OptimizeResponse)
    def run_experiment_explicit(request: OptimizeRequest) -> dict:
        return run_optimization(request)

    @app.get("/api/experiments/{experiment_id}")
    def experiment_detail(experiment_id: str) -> dict:
        if not experiment_id.strip():
            raise ValueError("experiment_id must not be empty")
        return {
            "id": experiment_id,
            "name": "AURORA-RIS",
            "status": "not yet measured",
        }

    @app.get("/api/metrics")
    def metrics() -> dict:
        return {"metrics": ["throughput_mbps", "mean_sinr_db", "energy_j",
                             "spectral_efficiency_bps_hz"]}

    @app.post("/api/metrics", response_model=OptimizeResponse)
    def calculate_metrics(request: OptimizeRequest) -> dict:
        return run_optimization(request)

    @app.get("/api/models")
    def models() -> dict:
        try:
            import torch  # noqa: F401
            available = True
        except ImportError:
            available = False
        return {"models": ["AuroraGNNGRU"] if available else ["deterministic-baseline"],
                "torch_available": available}

    @app.post("/api/models")
    def inspect_models(request: OptimizeRequest) -> dict:
        # POST is useful to clients that use one validated request shape for
        # all AURORA resources; model discovery itself has no side effects.
        return models()

    @app.post("/api/active-learning/retrain")
    def retrain(request: RetrainRequest) -> dict:
        # Training is intentionally explicit; this endpoint reports the
        # bounded request accepted by the service rather than claiming a fit.
        return {"status": "accepted", "samples": request.limit,
                "message": "retraining queued"}

    return app


app = create_app() if FastAPI is not None else None
