"""Bounded active-learning buffer for uncertain observations."""
from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import Any


@dataclass(frozen=True)
class Observation:
    features: Any
    action: Any
    reward: float
    uncertainty: float
    reason: str = ""


class ActiveLearningBuffer:
    def __init__(self, capacity: int = 1024, uncertainty_threshold: float = 0.25) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity, self.uncertainty_threshold = capacity, float(uncertainty_threshold)
        self._items: deque[Observation] = deque(maxlen=capacity)

    def add(self, observation: Observation) -> bool:
        if observation.uncertainty < self.uncertainty_threshold:
            return False
        self._items.append(observation)
        return True

    def __len__(self) -> int:
        return len(self._items)

    def sample(self, limit: int | None = None) -> list[Observation]:
        items = list(self._items)
        return items[-limit:] if limit else items
