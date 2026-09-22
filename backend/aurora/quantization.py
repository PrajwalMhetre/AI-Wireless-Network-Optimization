"""RIS phase quantization utilities."""
from __future__ import annotations
import numpy as np


def quantize_phases(phases: np.ndarray, bits: int = 2) -> np.ndarray:
    if int(bits) <= 0:
        raise ValueError("bits must be positive")
    x = np.asarray(phases, dtype=float)
    levels = 2 ** int(bits)
    step = 2 * np.pi / levels
    return np.mod(np.round(x / step) * step, 2 * np.pi)


def phase_codebook(bits: int) -> np.ndarray:
    if bits <= 0:
        raise ValueError("bits must be positive")
    return np.linspace(0.0, 2 * np.pi, 2 ** bits, endpoint=False)


quantize_ris_phases = quantize_phases
