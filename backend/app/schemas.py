"""Pydantic request and response schemas."""
from __future__ import annotations
from typing import Any

try:
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover
    BaseModel = object
    def Field(default=None, **kwargs): return default


class OptimizeRequest(BaseModel):
    seed: int = Field(default=42, ge=0)
    num_users: int = Field(default=4, ge=1, le=256)
    num_elements: int = Field(default=16, ge=1, le=4096)
    num_channels: int = Field(default=8, ge=1, le=128)
    ris_phase_bits: int = Field(default=2, ge=1, le=8)
    phase_resolution: int | None = Field(default=None, ge=1, le=8)
    snr_db: float = Field(default=20.0, ge=-100.0, le=100.0)
    snr: float | None = Field(default=None, ge=-100.0, le=100.0)
    csi_error: float = Field(default=0.0, ge=0.0, le=1.0)
    csi_error_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    mobility: float = Field(default=0.0, ge=0.0, le=1000.0)
    mobility_mps: float | None = Field(default=None, ge=0.0, le=1000.0)
    environment: str = Field(default="urban", min_length=1, max_length=64)
    confidence_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    ood_threshold: float = Field(default=0.5, gt=0.0)
    max_ood_score: float | None = Field(default=None, gt=0.0)


class OptimizeResponse(BaseModel):
    channels: list[int]
    phases: list[float]
    accepted: bool
    gate_reason: str
    confidence: float
    ood_score: float
    simulation: dict[str, Any]
    metrics: dict[str, float]
    optimizer_metadata: dict[str, Any] = Field(default_factory=dict)
    active_learning_reason: str | None = None
    uncertainty: float = 0.0
    experiment_id: str = ""
    scenario: dict[str, Any] = Field(default_factory=dict)
    prediction: dict[str, Any] = Field(default_factory=dict)
    gate: dict[str, Any] = Field(default_factory=dict)
    decision: dict[str, Any] = Field(default_factory=dict)
    optimizer: dict[str, Any] = Field(default_factory=dict)
    ris: dict[str, Any] = Field(default_factory=dict)
    comparison: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    simulator: str
    torch_available: bool


class RetrainRequest(BaseModel):
    limit: int = Field(default=128, ge=1, le=10000)
