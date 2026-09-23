"""Classical coordinate refinement used after a learned proposal."""
from __future__ import annotations
from typing import Callable
import numpy as np


def coordinate_refine(
    initial: np.ndarray,
    objective: Callable[[np.ndarray], float],
    levels: np.ndarray | None = None,
    max_rounds: int = 3,
    stats: dict[str, int] | None = None,
) -> np.ndarray:
    x = np.asarray(initial, dtype=float).copy()
    candidates = np.asarray(levels if levels is not None else [0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
    if stats is None:
        stats = {}
    stats.setdefault("objective_evaluations", 0)
    stats.setdefault("iterations", 0)

    def tracked_objective(proposal: np.ndarray) -> float:
        score = float(objective(proposal))
        stats["objective_evaluations"] = int(stats.get("objective_evaluations", 0)) + 1
        return score

    best = float(tracked_objective(x))
    for _ in range(max(1, max_rounds)):
        stats["iterations"] = int(stats.get("iterations", 0)) + 1
        changed = False
        for i in range(len(x)):
            local_best, local_value = x[i], best
            for value in candidates:
                proposal = x.copy()
                proposal[i] = value
                score = float(tracked_objective(proposal))
                if score > local_value + 1e-12:
                    local_best, local_value = value, score
            if local_best != x[i]:
                x[i], best, changed = local_best, local_value, True
        if not changed:
            break
    return x


coordinate_descent = coordinate_refine
