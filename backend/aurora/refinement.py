"""Classical coordinate refinement used after a learned proposal."""
from __future__ import annotations
from typing import Callable
import numpy as np


def coordinate_refine(initial: np.ndarray, objective: Callable[[np.ndarray], float],
                      levels: np.ndarray | None = None, max_rounds: int = 3) -> np.ndarray:
    x = np.asarray(initial, dtype=float).copy()
    candidates = np.asarray(levels if levels is not None else [0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
    best = float(objective(x))
    for _ in range(max(1, max_rounds)):
        changed = False
        for i in range(len(x)):
            local_best, local_value = x[i], best
            for value in candidates:
                proposal = x.copy()
                proposal[i] = value
                score = float(objective(proposal))
                if score > local_value + 1e-12:
                    local_best, local_value = value, score
            if local_best != x[i]:
                x[i], best, changed = local_best, local_value, True
        if not changed:
            break
    return x


coordinate_descent = coordinate_refine
