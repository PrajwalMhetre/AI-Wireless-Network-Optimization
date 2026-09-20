"""Simulation data collection metadata helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class SimulationRunMetadata:
    """Metadata recorded beside generated NS-3 outputs."""

    scenario_name: str
    random_seed: int
    duration_s: float
    status: str = "experiment_pending"


def write_run_manifest(path: str | Path, metadata: SimulationRunMetadata) -> None:
    """Write simulation run metadata as JSON."""

    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(asdict(metadata), indent=2), encoding="utf-8")

