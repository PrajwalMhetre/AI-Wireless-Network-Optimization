"""Classical baselines for comparisons."""
from __future__ import annotations
import numpy as np


def lowest_interference(interference_dbm: np.ndarray) -> np.ndarray:
    x = np.asarray(interference_dbm)
    if x.ndim != 2:
        raise ValueError("interference must be users by channels")
    return np.argmin(x, axis=1)


def random_policy(num_users: int, num_channels: int, seed: int = 42) -> np.ndarray:
    return np.random.default_rng(seed).integers(num_channels, size=num_users)
