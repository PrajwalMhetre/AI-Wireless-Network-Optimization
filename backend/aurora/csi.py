"""CSI feature extraction and temporal windows."""
from __future__ import annotations
import numpy as np


def normalize_csi(csi: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(csi)
    if x.ndim < 2:
        raise ValueError("CSI must have at least two dimensions")
    scale = np.sqrt(np.mean(np.abs(x) ** 2, axis=tuple(range(1, x.ndim)), keepdims=True))
    return x / np.maximum(scale, eps)


def csi_features(csi: np.ndarray) -> np.ndarray:
    """Convert complex CSI to stable real features (magnitude, phase, statistics)."""
    x = normalize_csi(np.asarray(csi))
    magnitude, phase = np.abs(x), np.angle(x)
    return np.concatenate([magnitude.real, phase.real], axis=-1)


extract_csi_features = csi_features


def temporal_window(samples: np.ndarray, length: int = 8) -> np.ndarray:
    x = np.asarray(samples)
    if length <= 0 or x.shape[0] < length:
        raise ValueError("window length must be positive and fit the sample count")
    return np.stack([x[i - length + 1:i + 1] for i in range(length - 1, len(x))])
