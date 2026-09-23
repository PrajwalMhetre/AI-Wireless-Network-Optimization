"""Confidence/OOD safety gate for learned actions."""
from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GateDecision:
    accept: bool
    reason: str
    confidence: float
    ood_score: float


class ConfidenceGate:
    def __init__(self, min_confidence: float = 0.75, max_ood_score: float = 0.5) -> None:
        if not 0 <= min_confidence <= 1 or max_ood_score <= 0:
            raise ValueError("invalid gate thresholds")
        self.min_confidence, self.max_ood_score = min_confidence, max_ood_score

    def decide(self, confidence: float, ood_score: float) -> GateDecision:
        confidence, ood_score = float(confidence), float(ood_score)
        if not math.isfinite(confidence) or not math.isfinite(ood_score):
            return GateDecision(False, "invalid_model_output", confidence, ood_score)
        if confidence < self.min_confidence and ood_score > self.max_ood_score:
            return GateDecision(False, "low_confidence_and_high_ood", confidence, ood_score)
        if confidence < self.min_confidence:
            return GateDecision(False, "low_confidence", confidence, ood_score)
        if ood_score > self.max_ood_score:
            return GateDecision(False, "out_of_distribution", confidence, ood_score)
        return GateDecision(True, "accepted", confidence, ood_score)
