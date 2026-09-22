"""Deterministic NumPy simulator with an optional Sionna integration seam."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .scenario import Scenario


@dataclass(frozen=True)
class SimulationResult:
    throughput_mbps: float
    per_user_throughput_mbps: np.ndarray
    sinr_db: np.ndarray
    energy_j: float
    spectral_efficiency: float

    def to_dict(self) -> dict:
        return {
            "throughput_mbps": float(self.throughput_mbps),
            "per_user_throughput_mbps": self.per_user_throughput_mbps.tolist(),
            "sinr_db": self.sinr_db.tolist(),
            "energy_j": float(self.energy_j),
            "spectral_efficiency": float(self.spectral_efficiency),
        }


class NumpyWirelessSimulator:
    """Small deterministic link-budget simulator suitable for CI and demos."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = int(seed)

    def run(self, scenario: Scenario, channel: int | np.ndarray,
            phases: np.ndarray | None = None) -> SimulationResult:
        cfg = scenario.config
        ch = np.asarray(channel, dtype=int)
        if ch.ndim == 0:
            ch = np.full(cfg.num_users, int(ch))
        if ch.shape != (cfg.num_users,):
            raise ValueError("channel must be a scalar or one channel per user")
        if np.any((ch < 0) | (ch >= cfg.num_channels)):
            raise ValueError("channel index out of range")
        gains = scenario.channel_gains[np.arange(cfg.num_users), ch]
        if phases is not None:
            phase = np.asarray(phases)
            if phase.size != cfg.num_elements:
                raise ValueError("phase vector has wrong size")
            coherent_gain = float(np.abs(np.exp(1j * phase).mean()) ** 2 * cfg.num_elements)
        else:
            coherent_gain = 1.0
        signal = gains.mean(axis=1) * coherent_gain
        noise = 10 ** (scenario.interference_dbm[np.arange(cfg.num_users), ch] / 10)
        # Explicit SNR is an additional receiver-noise constraint, rather than
        # a metadata-only field.  CSI error reduces usable coherent signal.
        snr_noise = signal / np.maximum(10 ** (cfg.snr_db / 10), 1e-12)
        csi_penalty = 1.0 + float(cfg.csi_error)
        sinr = signal / (noise + snr_noise + csi_penalty * 10 ** (cfg.noise_power_dbm / 10))
        sinr_db = 10 * np.log10(np.maximum(sinr, 1e-12))
        rates = cfg.bandwidth_hz * np.log2(1.0 + sinr) / 1e6
        rates = np.minimum(rates, scenario.traffic_demand_mbps)
        return SimulationResult(float(rates.sum()), rates, sinr_db,
                                float(rates.sum() * 0.001),
                                float(rates.sum() / (cfg.bandwidth_hz / 1e6)))


class SionnaWirelessSimulator:
    """Optional integration point; importing the core never imports Sionna."""

    def __init__(self, *args, **kwargs) -> None:
        try:
            import sionna  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Sionna is optional; install it separately or use the NumPy simulator"
            ) from exc
        raise NotImplementedError(
            "Connect a Sionna scene to the Scenario interface before using this adapter"
        )


def make_simulator(kind: str = "numpy", seed: int = 42) -> NumpyWirelessSimulator:
    """Return the requested simulator, keeping Sionna an explicit opt-in."""
    if kind.lower() not in {"numpy", "sionna"}:
        raise ValueError("kind must be 'numpy' or 'sionna'")
    if kind.lower() == "sionna":
        return SionnaWirelessSimulator()  # type: ignore[return-value]
    return NumpyWirelessSimulator(seed)
