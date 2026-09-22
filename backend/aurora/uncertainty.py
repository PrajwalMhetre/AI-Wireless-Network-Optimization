"""Defensible uncertainty and OOD scores using ensembles and Mahalanobis distance."""
from __future__ import annotations
import numpy as np


def ensemble_uncertainty(predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(predictions, dtype=float)
    if p.ndim < 2 or p.shape[0] < 2:
        raise ValueError("predictions require at least two ensemble members")
    return p.mean(axis=0), p.var(axis=0)


class MahalanobisOOD:
    def __init__(self, reference: np.ndarray, regularization: float = 1e-5) -> None:
        x = np.asarray(reference, dtype=float)
        if x.ndim != 2 or len(x) < 2:
            raise ValueError("reference must contain at least two feature rows")
        self.mean = x.mean(axis=0)
        cov = np.cov(x, rowvar=False)
        cov = np.atleast_2d(cov) + regularization * np.eye(x.shape[1])
        self.inverse_covariance = np.linalg.pinv(cov)

    def score(self, x: np.ndarray) -> np.ndarray:
        delta = np.asarray(x, dtype=float) - self.mean
        return np.einsum("...i,ij,...j->...", delta, self.inverse_covariance, delta)
