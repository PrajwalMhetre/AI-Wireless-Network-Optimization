"""Scenario generation and validation for reproducible experiments."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping
import numpy as np


@dataclass(frozen=True)
class ScenarioConfig:
    num_users: int = 4
    num_elements: int = 16
    num_channels: int = 8
    antennas: int = 4
    seed: int = 42
    bandwidth_hz: float = 20e6
    noise_power_dbm: float = -90.0
    ris_phase_bits: int = 2
    # Dynamic conditions used by both CSI generation and the link budget.
    snr_db: float = 20.0
    snr: float | None = None
    csi_error: float = 0.0
    csi_error_rate: float | None = None
    mobility: float = 0.0
    mobility_mps: float | None = None
    environment: str = "urban"
    # ``phase_resolution`` is the public spec name; ris_phase_bits is retained
    # for backwards compatibility with existing experiment files.
    phase_resolution: int | None = None

    def __post_init__(self) -> None:
        if self.snr is not None:
            object.__setattr__(self, "snr_db", float(self.snr))
        if self.csi_error_rate is not None:
            object.__setattr__(self, "csi_error", float(self.csi_error_rate))
        if self.mobility_mps is not None:
            object.__setattr__(self, "mobility", float(self.mobility_mps))
        for name in ("num_users", "num_elements", "num_channels", "antennas", "ris_phase_bits"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.phase_resolution is not None:
            if int(self.phase_resolution) <= 0:
                raise ValueError("phase_resolution must be positive")
            object.__setattr__(self, "ris_phase_bits", int(self.phase_resolution))
        object.__setattr__(self, "phase_resolution", int(self.ris_phase_bits))
        if not 0.0 <= float(self.csi_error) <= 1.0:
            raise ValueError("csi_error must be between 0 and 1")
        if float(self.mobility) < 0:
            raise ValueError("mobility must be non-negative")
        if not str(self.environment).strip():
            raise ValueError("environment must not be empty")


@dataclass
class Scenario:
    """A complete state used by the simulator and optimizer."""

    config: ScenarioConfig
    user_positions: np.ndarray
    channel_gains: np.ndarray
    interference_dbm: np.ndarray
    traffic_demand_mbps: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def generate(cls, config: ScenarioConfig | None = None) -> "Scenario":
        cfg = config or ScenarioConfig()
        rng = np.random.default_rng(cfg.seed)
        positions = rng.uniform([-50.0, -50.0, 1.0], [50.0, 50.0, 2.0],
                                size=(cfg.num_users, 3))
        # Mobility is represented as a deterministic displacement during the
        # observation window, making it affect both topology and CSI.
        if cfg.mobility:
            positions[:, :2] += rng.normal(0.0, float(cfg.mobility), size=(cfg.num_users, 2))
        # Positive gain proxies: path loss, log-normal shadowing, and fading.
        distances = np.linalg.norm(positions - np.array([0.0, 0.0, 8.0]), axis=1)
        environment_factor = {
            "urban": 1.0, "indoor": 1.15, "rural": 0.82, "suburban": 0.92,
        }.get(str(cfg.environment).lower(), 1.0)
        path_loss = environment_factor * np.maximum(distances, 1.0) ** -2.2
        fading = rng.lognormal(mean=0.0, sigma=0.25,
                               size=(cfg.num_users, cfg.num_channels, cfg.antennas))
        gains = path_loss[:, None, None] * fading
        if cfg.csi_error:
            gains *= np.maximum(
                0.0, 1.0 + rng.normal(0.0, float(cfg.csi_error),
                                      size=gains.shape)
            )
        interference = rng.uniform(-95.0, -65.0,
                                   size=(cfg.num_users, cfg.num_channels))
        demand = rng.uniform(1.0, 20.0, size=cfg.num_users)
        return cls(cfg, positions, gains, interference, demand,
                   {"simulator": "numpy-fallback", "seed": cfg.seed,
                    "environment": cfg.environment, "mobility": cfg.mobility,
                    "snr_db": cfg.snr_db, "csi_error": cfg.csi_error})

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.__dict__.copy(),
            "user_positions": self.user_positions.tolist(),
            "interference_dbm": self.interference_dbm.tolist(),
            "traffic_demand_mbps": self.traffic_demand_mbps.tolist(),
            "metadata": self.metadata.copy(),
        }


def scenario_from_mapping(data: Mapping[str, Any]) -> Scenario:
    """Construct a scenario from JSON-compatible API input."""
    cfg_data = dict(data.get("config", {}))
    # Accept the spec spelling while keeping old JSON files valid.
    if "phase_resolution" in cfg_data and "ris_phase_bits" not in cfg_data:
        cfg_data["ris_phase_bits"] = cfg_data["phase_resolution"]
    cfg = ScenarioConfig(**cfg_data)
    if "channel_gains" not in data:
        return Scenario.generate(cfg)
    scenario = Scenario(cfg, np.asarray(data["user_positions"], dtype=float),
                        np.asarray(data["channel_gains"], dtype=float),
                        np.asarray(data["interference_dbm"], dtype=float),
                        np.asarray(data["traffic_demand_mbps"], dtype=float),
                        dict(data.get("metadata", {})))
    if scenario.user_positions.shape != (cfg.num_users, 3):
        raise ValueError("user_positions must have shape (num_users, 3)")
    if scenario.channel_gains.shape != (cfg.num_users, cfg.num_channels, cfg.antennas):
        raise ValueError("channel_gains has an invalid shape")
    if scenario.interference_dbm.shape != (cfg.num_users, cfg.num_channels):
        raise ValueError("interference_dbm has an invalid shape")
    if scenario.traffic_demand_mbps.shape != (cfg.num_users,):
        raise ValueError("traffic_demand_mbps has an invalid shape")
    return scenario
