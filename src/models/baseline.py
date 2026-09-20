"""Deterministic baseline policies for wireless optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ChannelState:
    """State observed for one wireless channel."""

    channel_id: int
    interference_dbm: float
    snr_db: float


class LowestInterferenceChannelSelector:
    """Rule-based baseline for reproducible channel selection.

    The policy chooses the channel with the lowest interference. Ties are
    broken by higher SNR and then lower channel ID.
    """

    def select_channel(self, channels: Sequence[ChannelState | Mapping[str, float]]) -> int:
        if not channels:
            raise ValueError("channels cannot be empty")

        normalized = [_to_channel_state(channel) for channel in channels]
        selected = min(
            normalized,
            key=lambda channel: (
                channel.interference_dbm,
                -channel.snr_db,
                channel.channel_id,
            ),
        )
        return selected.channel_id


def _to_channel_state(channel: ChannelState | Mapping[str, float]) -> ChannelState:
    if isinstance(channel, ChannelState):
        return channel
    return ChannelState(
        channel_id=int(channel["channel_id"]),
        interference_dbm=float(channel["interference_dbm"]),
        snr_db=float(channel["snr_db"]),
    )

