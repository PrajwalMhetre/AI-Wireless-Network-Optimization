"""Torch-only GNN + GRU policy model (no torch-geometric required)."""
from __future__ import annotations
from typing import Optional

try:
    import torch
    from torch import nn
except ImportError:  # keep non-ML API usable in minimal installs
    torch = None
    nn = object


if torch is not None:
    class GraphConv(nn.Module):
        def __init__(self, in_features: int, out_features: int) -> None:
            super().__init__()
            self.linear = nn.Linear(in_features, out_features)

        def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
            return self.linear(torch.bmm(adjacency, x))


    class AuroraGNNGRU(nn.Module):
        """Per-step graph encoder followed by temporal GRU and action heads."""
        def __init__(self, node_features: int, hidden: int = 64, actions: int = 8,
                     layers: int = 2) -> None:
            super().__init__()
            self.conv1 = GraphConv(node_features, hidden)
            self.conv2 = GraphConv(hidden, hidden)
            self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True)
            self.policy = nn.Linear(hidden, actions)
            self.value = nn.Linear(hidden, 1)
            self.log_variance = nn.Linear(hidden, 1)

        def forward(self, x: torch.Tensor, adjacency: torch.Tensor):
            # x: batch,time,nodes,features; adjacency: batch,nodes,nodes
            b, t, n, _ = x.shape
            a = adjacency.unsqueeze(1).expand(-1, t, -1, -1).reshape(b * t, n, n)
            h = x.reshape(b * t, n, -1)
            h = torch.relu(self.conv1(h, a))
            h = torch.relu(self.conv2(h, a)).mean(dim=1).reshape(b, t, -1)
            h, _ = self.gru(h)
            last = h[:, -1]
            return self.policy(last), self.value(last), self.log_variance(last)
else:
    class AuroraGNNGRU:  # pragma: no cover - import-time compatibility
        def __init__(self, *args, **kwargs):
            raise ImportError("AuroraGNNGRU requires the optional 'torch' dependency")
