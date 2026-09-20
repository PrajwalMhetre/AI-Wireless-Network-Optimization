"""Feature engineering helpers for wireless-network datasets."""

from __future__ import annotations


def estimate_rssi_dbm(tx_power_dbm: float, path_loss_db: float) -> float:
    """Estimate RSSI from transmit power and path loss."""

    return tx_power_dbm - path_loss_db


def normalize_load(packet_rate: float, max_packet_rate: float) -> float:
    """Normalize offered packet rate to the range [0, 1]."""

    if max_packet_rate <= 0:
        raise ValueError("max_packet_rate must be positive")
    value = packet_rate / max_packet_rate
    return max(0.0, min(1.0, value))

