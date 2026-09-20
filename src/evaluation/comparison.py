"""Helpers for AI vs conventional baseline comparisons."""

from __future__ import annotations


def percentage_change(candidate: float, baseline: float, higher_is_better: bool = True) -> float:
    """Return signed percentage improvement of candidate relative to baseline."""

    if baseline == 0:
        raise ValueError("baseline cannot be zero")

    change = ((candidate - baseline) / abs(baseline)) * 100.0
    return change if higher_is_better else -change

