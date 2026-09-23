"""
AURORA-RIS temporal sequence dataset.

Converts consecutive feature/target samples into
sequences suitable for the GNN + GRU model.
"""

from __future__ import annotations

import numpy as np


def create_temporal_sequences(
    X: np.ndarray,
    y: np.ndarray,
    sequence_length: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create sliding temporal sequences.
    """

    X = np.asarray(X)
    y = np.asarray(y)

    if sequence_length <= 0:
        raise ValueError("sequence_length must be greater than zero")

    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples")

    if len(X) < sequence_length:
        raise ValueError(
            "Not enough samples for the requested sequence length"
        )

    sequence_X = []
    sequence_y = []

    for start_index in range(len(X) - sequence_length + 1):
        end_index = start_index + sequence_length

        sequence_X.append(X[start_index:end_index])
        sequence_y.append(y[end_index - 1])

    return (
        np.stack(sequence_X).astype(np.float32),
        np.stack(sequence_y).astype(np.float32),
    )