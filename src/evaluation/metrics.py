"""Wireless-network performance metric formulas."""

from __future__ import annotations

from collections.abc import Iterable


def throughput_bps(received_bits: float, observation_time_s: float) -> float:
    """Return throughput in bits per second."""

    _require_non_negative(received_bits, "received_bits")
    _require_positive(observation_time_s, "observation_time_s")
    return received_bits / observation_time_s


def packet_delivery_ratio(received_packets: int, transmitted_packets: int) -> float:
    """Return packet delivery ratio as received packets divided by sent packets."""

    _require_non_negative(received_packets, "received_packets")
    _require_positive(transmitted_packets, "transmitted_packets")
    if received_packets > transmitted_packets:
        raise ValueError("received_packets cannot exceed transmitted_packets")
    return received_packets / transmitted_packets


def bit_error_rate(bit_errors: int, total_bits: int) -> float:
    """Return bit error rate as erroneous bits divided by total bits."""

    _require_non_negative(bit_errors, "bit_errors")
    _require_positive(total_bits, "total_bits")
    if bit_errors > total_bits:
        raise ValueError("bit_errors cannot exceed total_bits")
    return bit_errors / total_bits


def mean_latency_s(latencies_s: Iterable[float]) -> float:
    """Return arithmetic mean end-to-end latency in seconds."""

    values = list(latencies_s)
    if not values:
        raise ValueError("latencies_s cannot be empty")
    for value in values:
        _require_non_negative(value, "latency")
    return sum(values) / len(values)


def energy_consumption_j(power_w: float, duration_s: float) -> float:
    """Return energy consumption in joules."""

    _require_non_negative(power_w, "power_w")
    _require_non_negative(duration_s, "duration_s")
    return power_w * duration_s


def spectrum_efficiency_bps_per_hz(throughput: float, bandwidth_hz: float) -> float:
    """Return spectrum efficiency in bps/Hz."""

    _require_non_negative(throughput, "throughput")
    _require_positive(bandwidth_hz, "bandwidth_hz")
    return throughput / bandwidth_hz


def _require_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_non_negative(value: float, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")

