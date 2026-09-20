"""Optimization decision orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class OptimizationDecision:
    """A decision that can be applied to a wireless simulation."""

    action: str
    selected_channel: int | None = None
    tx_power_dbm: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ChannelSelectionOptimizer:
    """Use a supplied policy or model to choose a wireless channel."""

    def __init__(self, policy: Any) -> None:
        self.policy = policy

    def decide(self, network_state: Mapping[str, Any]) -> OptimizationDecision:
        channels = network_state.get("channels")
        if channels is None:
            raise ValueError("network_state must contain 'channels'")

        selected_channel = int(self.policy.select_channel(channels))
        return OptimizationDecision(
            action="channel_selection",
            selected_channel=selected_channel,
            metadata={"policy": self.policy.__class__.__name__},
        )

