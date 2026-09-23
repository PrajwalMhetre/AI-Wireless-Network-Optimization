"""
AURORA-RIS GNN + GRU training pipeline.

Trains the existing backend.aurora.AuroraGNNGRU model using
CSI feature sequences and classical optimization targets.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from backend.aurora.graph import build_knn_graph
from backend.aurora.model import AuroraGNNGRU
from src.data.training_dataset import generate_optimized_dataset
from src.data.sequence_dataset import create_temporal_sequences


SEED = 42
SEQUENCE_LENGTH = 5
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.2

RESULTS_DIR = "results"
CHECKPOINT_PATH = os.path.join(
    RESULTS_DIR,
    "best_aurora_gnn_gru.pt",
)


def set_seed(seed: int = SEED) -> None:
    """Make training as reproducible as possible."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_adjacency(
    X: np.ndarray,
) -> np.ndarray:
    """
    Build one KNN adjacency matrix for each temporal sequence.

    The graph is constructed from the final timestep because the
    model uses one adjacency matrix across the temporal window.
    """

    adjacency_matrices = []

    for sequence in X:
        final_features = sequence[-1]

        graph = build_knn_graph(
            final_features,
            k=min(
                3,
                max(0, len(final_features) - 1),
            ),
        )

        adjacency_matrices.append(
            graph.normalized_adjacency().astype(np.float32)
        )

    return np.stack(
        adjacency_matrices
    ).astype(np.float32)


def prepare_data():
    """Generate dataset and convert it into temporal sequences."""

    X, channel_y, phase_y, samples = (
        generate_optimized_dataset()
    )

    print("Generated dataset")
    print("-----------------")
    print(f"X shape: {X.shape}")
    print(f"Channel target shape: {channel_y.shape}")
    print(f"Phase target shape: {phase_y.shape}")
    print(f"Samples: {len(samples)}")
    print()

    # The existing sequence utility accepts one target array.
    # We create temporal sequences from X and use the final
    # timestep targets for channel and phase supervision.
    sequence_X, _ = create_temporal_sequences(
        X,
        phase_y,
        sequence_length=SEQUENCE_LENGTH,
    )

    target_start = SEQUENCE_LENGTH - 1

    sequence_channel_y = channel_y[
        target_start:
    ]

    sequence_phase_y = phase_y[
        target_start:
    ]

    adjacency = build_adjacency(
        sequence_X
    )

    return (
        sequence_X,
        adjacency,
        sequence_channel_y,
        sequence_phase_y,
    )


def split_data(
    X: np.ndarray,
    adjacency: np.ndarray,
    channel_y: np.ndarray,
    phase_y: np.ndarray,
):
    """Create deterministic train/validation splits."""

    total = len(X)

    split_index = int(
        total * (1.0 - VALIDATION_SPLIT)
    )

    if split_index <= 0 or split_index >= total:
        raise ValueError(
            "Invalid train/validation split"
        )

    train_slice = slice(
        0,
        split_index,
    )

    validation_slice = slice(
        split_index,
        None,
    )

    return (
        X[train_slice],
        adjacency[train_slice],
        channel_y[train_slice],
        phase_y[train_slice],
        X[validation_slice],
        adjacency[validation_slice],
        channel_y[validation_slice],
        phase_y[validation_slice],
    )


def make_loader(
    X: np.ndarray,
    adjacency: np.ndarray,
    channel_y: np.ndarray,
    phase_y: np.ndarray,
    shuffle: bool,
):
    """Create a PyTorch DataLoader."""

    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(adjacency),
        torch.from_numpy(channel_y),
        torch.from_numpy(phase_y),
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
    )


def calculate_loss(
    model: AuroraGNNGRU,
    batch_X: torch.Tensor,
    batch_adjacency: torch.Tensor,
    batch_channels: torch.Tensor,
    batch_phases: torch.Tensor,
):
    """Calculate channel, phase and uncertainty-aware training loss."""

    policy, value, log_variance = model(
        batch_X,
        batch_adjacency,
    )

    num_channels = model.policy.out_features - (
        batch_phases.shape[1]
    )

    channel_logits = policy[
        :, :num_channels
    ]

    phase_logits = policy[
        :, num_channels:
    ]

    # The current controller selects one common channel
    # for all users, so the first user's channel represents
    # the common optimized channel target.
    channel_target = batch_channels[
        :, 0
    ].long()

    channel_loss = nn.functional.cross_entropy(
        channel_logits,
        channel_target,
    )

    # Controller converts phase logits through sigmoid to
    # [0, 2*pi]. Therefore train against normalized phases.
    normalized_phase_target = (
        batch_phases / (2.0 * np.pi)
    )

    predicted_phase = torch.sigmoid(
        phase_logits
    )

    phase_loss = nn.functional.mse_loss(
        predicted_phase,
        normalized_phase_target,
    )

    # The target throughput is not directly available inside
    # this batch, so value/log-variance heads are regularized
    # lightly instead of inventing a reward target.
    variance_regularization = (
        log_variance ** 2
    ).mean()

    value_regularization = (
        value ** 2
    ).mean()

    loss = (
        channel_loss
        + phase_loss
        + 0.01 * variance_regularization
        + 0.001 * value_regularization
    )

    return (
        loss,
        channel_loss,
        phase_loss,
    )


def evaluate(
    model: AuroraGNNGRU,
    loader: DataLoader,
):
    """Evaluate model on validation data."""

    model.eval()

    total_loss = 0.0
    total_channel_loss = 0.0
    total_phase_loss = 0.0
    batches = 0

    with torch.no_grad():

        for (
            batch_X,
            batch_adjacency,
            batch_channels,
            batch_phases,
        ) in loader:

            (
                loss,
                channel_loss,
                phase_loss,
            ) = calculate_loss(
                model,
                batch_X,
                batch_adjacency,
                batch_channels,
                batch_phases,
            )

            total_loss += float(loss.item())
            total_channel_loss += float(
                channel_loss.item()
            )
            total_phase_loss += float(
                phase_loss.item()
            )

            batches += 1

    if batches == 0:
        return 0.0, 0.0, 0.0

    return (
        total_loss / batches,
        total_channel_loss / batches,
        total_phase_loss / batches,
    )


def main():
    """Run the complete training pipeline."""

    set_seed()

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    (
        X,
        adjacency,
        channel_y,
        phase_y,
    ) = prepare_data()

    (
        train_X,
        train_adjacency,
        train_channels,
        train_phases,
        val_X,
        val_adjacency,
        val_channels,
        val_phases,
    ) = split_data(
        X,
        adjacency,
        channel_y,
        phase_y,
    )

    print("Temporal dataset")
    print("-----------------")
    print(f"Train sequences: {len(train_X)}")
    print(f"Validation sequences: {len(val_X)}")
    print(f"Sequence shape: {train_X.shape}")
    print(f"Adjacency shape: {train_adjacency.shape}")
    print()

    train_loader = make_loader(
        train_X,
        train_adjacency,
        train_channels,
        train_phases,
        shuffle=True,
    )

    validation_loader = make_loader(
        val_X,
        val_adjacency,
        val_channels,
        val_phases,
        shuffle=False,
    )

    node_features = train_X.shape[-1]
    num_channels = int(
        np.max(channel_y) + 1
    )
    num_elements = phase_y.shape[1]

    actions = (
        num_channels
        + num_elements
    )

    model = AuroraGNNGRU(
        node_features=node_features,
        actions=actions,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_validation_loss = float(
        "inf"
    )

    print("Training AURORA GNN + GRU")
    print("-------------------------")

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        running_loss = 0.0
        batches = 0

        for (
            batch_X,
            batch_adjacency,
            batch_channels,
            batch_phases,
        ) in train_loader:

            optimizer.zero_grad()

            (
                loss,
                _,
                _,
            ) = calculate_loss(
                model,
                batch_X,
                batch_adjacency,
                batch_channels,
                batch_phases,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            running_loss += float(
                loss.item()
            )

            batches += 1

        train_loss = (
            running_loss / batches
        )

        (
            validation_loss,
            validation_channel_loss,
            validation_phase_loss,
        ) = evaluate(
            model,
            validation_loader,
        )

        if validation_loss < best_validation_loss:

            best_validation_loss = (
                validation_loss
            )

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),
                    "node_features":
                        node_features,
                    "num_channels":
                        num_channels,
                    "num_elements":
                        num_elements,
                    "actions":
                        actions,
                    "sequence_length":
                        SEQUENCE_LENGTH,
                    "best_validation_loss":
                        best_validation_loss,
                },
                CHECKPOINT_PATH,
            )

            marker = "*"

        else:
            marker = ""

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train={train_loss:.4f} | "
            f"val={validation_loss:.4f} | "
            f"channel={validation_channel_loss:.4f} | "
            f"phase={validation_phase_loss:.4f} "
            f"{marker}"
        )

    print()
    print("Training complete")
    print("------------------")
    print(
        f"Best validation loss: "
        f"{best_validation_loss:.6f}"
    )
    print(
        f"Checkpoint: "
        f"{CHECKPOINT_PATH}"
    )


if __name__ == "__main__":
    main()