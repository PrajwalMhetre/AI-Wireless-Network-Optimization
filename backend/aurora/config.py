"""Configuration helpers for AURORA experiments."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class AuroraConfig:
    seed: int = 42
    simulator: str = "numpy"
    confidence_threshold: float = 0.6
    ood_threshold: float = 12.0
    phase_bits: int = 2
    users: int = 4
    elements: int = 16
    channels: int = 8

    @classmethod
    def from_file(cls, path: str | Path) -> "AuroraConfig":
        data = json.loads(Path(path).read_text())
        return cls(**data)
