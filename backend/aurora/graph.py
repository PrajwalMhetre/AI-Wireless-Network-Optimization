"""Graph construction without a torch-geometric dependency."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Graph:
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_weight: np.ndarray

    def normalized_adjacency(self) -> np.ndarray:
        n = len(self.node_features)
        a = np.zeros((n, n), dtype=float)
        a[self.edge_index[0], self.edge_index[1]] = self.edge_weight
        a += np.eye(n)
        d = np.maximum(a.sum(axis=1), 1e-12)
        return a / np.sqrt(d[:, None] * d[None, :])


def build_knn_graph(features: np.ndarray, k: int = 3) -> Graph:
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or not len(x):
        raise ValueError("features must be a non-empty 2D array")
    k = max(1, min(int(k), len(x) - 1)) if len(x) > 1 else 0
    dist = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
    edges, weights = [], []
    for i in range(len(x)):
        for j in np.argsort(dist[i])[1:k + 1]:
            edges.append((i, int(j)))
            weights.append(float(np.exp(-dist[i, j])))
    edge_index = np.asarray(edges, dtype=int).T if edges else np.empty((2, 0), dtype=int)
    return Graph(x, edge_index, np.asarray(weights, dtype=float))


def build_graph(features: np.ndarray, k: int = 3) -> Graph:
    """Descriptive alias used by experiment code."""
    return build_knn_graph(features, k)
