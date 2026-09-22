"""Metrics reported by AURORA experiments."""
from __future__ import annotations
import numpy as np


def compute_metrics(throughput_mbps: float, sinr_db: np.ndarray,
                    energy_j: float, bandwidth_hz: float,
                    delivered: int | None = None, sent: int | None = None) -> dict[str, float]:
    result = {
        "throughput_mbps": float(throughput_mbps),
        "mean_sinr_db": float(np.mean(sinr_db)),
        "energy_j": float(energy_j),
        "spectral_efficiency_bps_hz": float(throughput_mbps * 1e6 / bandwidth_hz),
    }
    if sent is not None:
        if sent <= 0 or delivered is None or not 0 <= delivered <= sent:
            raise ValueError("delivered/sent must satisfy 0 <= delivered <= sent and sent > 0")
        result["pdr"] = delivered / sent
    return result
