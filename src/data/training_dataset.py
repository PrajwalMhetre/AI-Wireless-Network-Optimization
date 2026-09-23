
"""
AURORA-RIS training dataset generation.

Uses the existing backend/aurora architecture to generate
CSI-based node features and classical optimization targets.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from backend.aurora.csi import csi_features
from backend.aurora.quantization import quantize_phases
from backend.aurora.refinement import coordinate_refine
from backend.aurora.scenario import Scenario, ScenarioConfig
from backend.aurora.simulator import NumpyWirelessSimulator


def _extract_features(
    scenario: Scenario,
) -> tuple[np.ndarray, np.ndarray]:

    gains = np.asarray(
        scenario.channel_gains,
        dtype=float,
    )

    phase_basis = np.linspace(
        0.0,
        np.pi,
        scenario.config.antennas,
        endpoint=False,
    )

    csi = gains.astype(complex) * np.exp(
        1j * phase_basis
    )[None, None, :]

    features = csi_features(csi).reshape(
        scenario.config.num_users,
        -1,
    )

    return (
        features.astype(np.float32),
        csi,
    )


def _optimize_target(
    scenario: Scenario,
    simulator: NumpyWirelessSimulator,
    max_rounds: int = 2,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Generate a classical optimization target.

    Selects the common channel using channel quality and
    then refines the RIS phase configuration.
    """

    cfg = scenario.config

    best_channel = None
    best_score = -np.inf

    gains = np.asarray(
        scenario.channel_gains,
        dtype=float,
    )

    interference_dbm = np.asarray(
        scenario.interference_dbm,
        dtype=float,
    )

    for channel in range(cfg.num_channels):

        signal = gains[
            :,
            channel,
            :,
        ].mean(axis=1)

        interference = 10.0 ** (
            interference_dbm[
                :,
                channel,
            ] / 10.0
        )

        noise = 10.0 ** (
            cfg.noise_power_dbm / 10.0
        )

        snr_noise = signal / max(
            10.0 ** (cfg.snr_db / 10.0),
            1e-12,
        )

        sinr = signal / (
            interference
            + snr_noise
            + noise
        )

        channel_score = float(
            np.sum(
                np.log2(
                    1.0 + np.maximum(
                        sinr,
                        1e-12,
                    )
                )
            )
        )

        if channel_score > best_score:
            best_score = channel_score

            best_channel = np.full(
                cfg.num_users,
                channel,
                dtype=int,
            )

    if best_channel is None:
        raise RuntimeError(
            "Unable to determine a channel target"
        )

    initial_phases = np.zeros(
        cfg.num_elements,
        dtype=float,
    )

    quantized_initial = quantize_phases(
        initial_phases,
        cfg.ris_phase_bits,
    )

    optimized_phases = coordinate_refine(
        quantized_initial,
        lambda phases: simulator.run(
            scenario,
            best_channel,
            phases,
        ).throughput_mbps,
        levels=np.linspace(
            0.0,
            2.0 * np.pi,
            2 ** cfg.ris_phase_bits,
            endpoint=False,
        ),
        max_rounds=max_rounds,
    )

    optimized_phases = quantize_phases(
        optimized_phases,
        cfg.ris_phase_bits,
    )

    final_result = simulator.run(
        scenario,
        best_channel,
        optimized_phases,
    )

    return (
        best_channel.astype(np.int64),
        optimized_phases.astype(np.float32),
        float(final_result.throughput_mbps),
    )


def generate_optimized_dataset(
    config: ScenarioConfig | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
]:

    if config is None:
        config = ScenarioConfig(
            num_users=4,
            num_elements=16,
            num_channels=8,
            antennas=4,
            seed=42,
        )

    feature_samples = []
    channel_targets = []
    phase_targets = []
    samples = []

    simulator = NumpyWirelessSimulator(
        seed=config.seed,
    )

    for sample_index in range(100):

        sample_config = ScenarioConfig(
            num_users=config.num_users,
            num_elements=config.num_elements,
            num_channels=config.num_channels,
            antennas=config.antennas,
            seed=config.seed + sample_index,
            bandwidth_hz=config.bandwidth_hz,
            noise_power_dbm=config.noise_power_dbm,
            ris_phase_bits=config.ris_phase_bits,
            snr_db=config.snr_db,
            csi_error=config.csi_error,
            mobility=config.mobility,
            environment=config.environment,
        )

        scenario = Scenario.generate(
            sample_config
        )

        features, _ = _extract_features(
            scenario
        )

        channels, phases, reward = _optimize_target(
            scenario,
            simulator,
        )

        feature_samples.append(features)
        channel_targets.append(channels)
        phase_targets.append(phases)

        samples.append(
            {
                "index": sample_index,
                "seed": sample_config.seed,
                "reward_throughput_mbps": reward,
                "config": sample_config.__dict__.copy(),
                "user_positions": scenario.user_positions.tolist(),
                "interference_dbm": scenario.interference_dbm.tolist(),
                "traffic_demand_mbps": scenario.traffic_demand_mbps.tolist(),
            }
        )

    X = np.stack(
        feature_samples
    ).astype(np.float32)

    channel_y = np.stack(
        channel_targets
    ).astype(np.int64)

    phase_y = np.stack(
        phase_targets
    ).astype(np.float32)

    return (
        X,
        channel_y,
        phase_y,
        samples,
    )


if __name__ == "__main__":

    config = ScenarioConfig(
        num_users=4,
        num_elements=8,
        num_channels=4,
        antennas=4,
        ris_phase_bits=2,
        seed=42,
    )

    X, channel_y, phase_y, samples = (
        generate_optimized_dataset(config)
    )

    print("AURORA-RIS Training Dataset")
    print("----------------------------")
    print(f"Samples: {len(samples)}")
    print(f"Feature shape: {X.shape}")
    print(f"Channel target shape: {channel_y.shape}")
    print(f"Phase target shape: {phase_y.shape}")
    print()
    print("Unique channel targets:")
    print(np.unique(channel_y))
    print()
    print("Channel counts:")
    print(np.bincount(
        channel_y.ravel(),
        minlength=config.num_channels,
    ))

