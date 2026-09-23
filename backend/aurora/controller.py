"""End-to-end AURORA CSI, graph, model and safe optimization controller."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import time

import numpy as np

from .active_learning import ActiveLearningBuffer, Observation
from .baselines import lowest_interference
from .csi import csi_features
from .gate import ConfidenceGate
from .graph import build_knn_graph
from .model import AuroraGNNGRU
from .quantization import quantize_phases
from .refinement import coordinate_refine
from .scenario import Scenario, ScenarioConfig, scenario_from_mapping
from .simulator import NumpyWirelessSimulator
from .uncertainty import MahalanobisOOD

try:  # torch is deliberately optional for the deterministic CI fallback.
    import torch
except ImportError:  # pragma: no cover - exercised in minimal installations
    torch = None


@dataclass
class OptimizationResult:
    channels: list[int]
    phases: list[float]
    accepted: bool
    gate_reason: str
    confidence: float
    ood_score: float
    simulation: dict[str, Any]
    metrics: dict[str, float]
    optimizer_metadata: dict[str, Any] = field(default_factory=dict)
    active_learning_reason: str | None = None
    uncertainty: float = 0.0
    experiment_id: str = ""
    scenario: dict[str, Any] = field(default_factory=dict)
    prediction: dict[str, Any] = field(default_factory=dict)
    gate: dict[str, Any] = field(default_factory=dict)
    decision: dict[str, Any] = field(default_factory=dict)
    optimizer: dict[str, Any] = field(default_factory=dict)
    ris: dict[str, Any] = field(default_factory=dict)
    comparison: dict[str, Any] = field(default_factory=dict)


class AURORAController:
    def __init__(
        self,
        scenario: Scenario | None = None,
        seed: int = 42,
        confidence_threshold: float = 0.6,
        ood_threshold: float = 12.0,
        max_ood_score: float | None = None,
    ) -> None:
        self.seed = int(seed)
        self.scenario = scenario or Scenario.generate(ScenarioConfig(seed=seed))
        self.simulator = NumpyWirelessSimulator(seed)
        self.gate = ConfidenceGate(
            confidence_threshold,
            float(max_ood_score if max_ood_score is not None else ood_threshold),
        )
        self.buffer = ActiveLearningBuffer()
        self.model: Any | None = None
        self._model_shape: tuple[int, int] | None = None
        self._ood_detector = self._make_ood_detector(self.scenario)

    @staticmethod
    def _feature_pipeline(state: Scenario) -> tuple[np.ndarray, Any, np.ndarray]:
        # Channel gains are the measured CSI amplitude.  A deterministic
        # phase basis preserves complex CSI semantics without extra state.
        gains = np.asarray(state.channel_gains, dtype=float)
        phase_basis = np.linspace(0.0, np.pi, gains.shape[-1], endpoint=False)
        csi = gains.astype(complex) * np.exp(1j * phase_basis)[None, None, :]
        features = csi_features(csi).reshape(state.config.num_users, -1)
        graph = build_knn_graph(features, k=min(3, max(0, len(features) - 1)))
        return features, graph, csi

    def _make_ood_detector(self, state: Scenario) -> MahalanobisOOD:
        features, _, csi = self._feature_pipeline(state)
        summary = self._feature_summary(features, csi)
        if len(summary) < 2:
            summary = np.concatenate([summary, summary + 1e-3], axis=0)
        return MahalanobisOOD(summary)

    @staticmethod
    def _feature_summary(features: np.ndarray, csi: np.ndarray | None = None) -> np.ndarray:
        summary = np.column_stack(
            (np.mean(features, axis=1), np.std(features, axis=1),
             np.linalg.norm(features, axis=1))
        )
        if csi is not None:
            # Normalized CSI is ideal for the policy, while absolute received
            # power remains useful for detecting distribution shift.
            summary = np.column_stack((summary, np.mean(np.abs(csi), axis=(1, 2))))
        return summary

    def _get_model(self, node_features: int, actions: int) -> Any:
        if torch is None:
            raise ImportError("torch is unavailable")
        shape = (int(node_features), int(actions))
        if self.model is None or self._model_shape != shape:
            torch.manual_seed(self.seed)
            self.model = AuroraGNNGRU(node_features, actions=actions)
            self.model.eval()
            self._model_shape = shape
        return self.model

    def _learned_proposal(
        self, features: np.ndarray, graph: Any, state: Scenario, csi: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float, float, dict[str, Any]]:
        """Run CSI -> graph -> GNN/GRU and derive scores from its outputs."""
        actions = state.config.num_channels + state.config.num_elements
        model = self._get_model(features.shape[1], actions)
        with torch.no_grad():
            x = torch.as_tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            adjacency = torch.as_tensor(
                graph.normalized_adjacency(), dtype=torch.float32
            ).unsqueeze(0)
            policy, value, log_variance = model(x, adjacency)
            policy_np = policy[0].detach().cpu().numpy()
            value_np = float(value[0, 0].detach().cpu())
            logvar_np = float(log_variance[0, 0].detach().cpu())

        channel_logits = policy_np[:state.config.num_channels]
        # Softmax confidence is a direct property of the policy output;
        # value/log-variance provide an independent uncertainty signal.
        shifted = channel_logits - np.max(channel_logits)
        probabilities = np.exp(shifted)
        probabilities /= np.maximum(probabilities.sum(), 1e-12)
        policy_confidence = float(np.max(probabilities))
        predictive_variance = float(np.exp(np.clip(logvar_np, -20.0, 20.0)))
        confidence = float(np.clip(
            policy_confidence * np.exp(-0.5 * predictive_variance), 0.0, 1.0
        ))
        ood_score = float(np.mean(self._ood_detector.score(
            self._feature_summary(features, csi)
        )))
        channels = np.full(
            state.config.num_users, int(np.argmax(channel_logits)), dtype=int
        )
        phase_logits = policy_np[state.config.num_channels:]
        if len(phase_logits) < state.config.num_elements:
            phase_logits = np.resize(phase_logits, state.config.num_elements)
        phases = np.mod(2.0 * np.pi * (1.0 / (1.0 + np.exp(-phase_logits))), 2.0 * np.pi)
        return channels, phases, confidence, ood_score, {
            "model": "AuroraGNNGRU",
            "model_value": value_np,
            "predictive_variance": predictive_variance,
            "policy_entropy": float(-np.sum(
                probabilities * np.log(np.maximum(probabilities, 1e-12))
            )),
        }

    def _fallback_proposal(
        self, state: Scenario, features: np.ndarray, csi: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float, float, dict[str, Any]]:
        channels = lowest_interference(state.interference_dbm)
        # Feature quality and distance from the reference distribution are
        # still measured, rather than presenting a fabricated confidence.
        spread = float(np.mean(np.std(features, axis=1)))
        confidence = float(np.clip(1.0 / (1.0 + spread), 0.0, 1.0))
        ood_score = float(np.mean(self._ood_detector.score(
            self._feature_summary(features, csi)
        )))
        phases = np.zeros(state.config.num_elements, dtype=float)
        return channels, phases, confidence, ood_score, {
            "model": "deterministic-baseline",
            "fallback_reason": "torch_unavailable",
        }

    def optimize(self, scenario: Scenario | dict[str, Any] | None = None) -> OptimizationResult:
        started_at = time.perf_counter()
        state = scenario_from_mapping(scenario) if isinstance(scenario, dict) else (
            scenario or self.scenario
        )
        features, graph, csi = self._feature_pipeline(state)
        if torch is None:
            channels, phases, confidence, ood_score, model_metadata = (
                self._fallback_proposal(state, features, csi)
            )
        else:
            # A torch installation is authoritative: model errors are surfaced
            # rather than silently changing the requested execution path.
            channels, phases, confidence, ood_score, model_metadata = (
                self._learned_proposal(features, graph, state, csi)
            )

        gate = self.gate.decide(confidence, ood_score)
        refinement_invoked = not gate.accept
        reasons = []
        if confidence < self.gate.min_confidence:
            reasons.append("low_confidence")
        if ood_score > self.gate.max_ood_score:
            reasons.append("out_of_distribution")
        active_reason = ";".join(reasons) if reasons else (None if gate.accept else gate.reason)
        pre_refinement_phases = quantize_phases(np.asarray(phases, dtype=float), state.config.ris_phase_bits)
        pre_refinement_channels = np.asarray(channels, dtype=int)
        pre_refinement_sim = self.simulator.run(state, pre_refinement_channels, pre_refinement_phases)
        optimizer_stats = {"objective_evaluations": 0, "iterations": 0}
        if refinement_invoked:
            # Unsafe learned actions are replaced by the conservative channel
            # baseline and classical phase refinement.
            channels = lowest_interference(state.interference_dbm)
            phases = coordinate_refine(
                quantize_phases(pre_refinement_phases, state.config.ris_phase_bits),
                lambda p: self._score(state, channels, p),
                stats=optimizer_stats,
            )
        else:
            phases = quantize_phases(pre_refinement_phases, state.config.ris_phase_bits)

        sim = self.simulator.run(state, channels, phases)
        uncertainty = float(np.clip(
            (1.0 - confidence) + min(max(ood_score, 0.0) /
                                     max(self.gate.max_ood_score, 1e-12), 1.0),
            0.0, 2.0,
        ))
        observation_reason = active_reason or (
            "uncertainty" if uncertainty >= self.buffer.uncertainty_threshold else "accepted"
        )
        buffered = self.buffer.add(Observation(
            features, channels.tolist(), sim.throughput_mbps, uncertainty,
            reason=observation_reason,
        ))
        if buffered and active_reason is None:
            active_reason = observation_reason
        metrics = {
            "throughput_mbps": sim.throughput_mbps,
            "mean_sinr_db": float(np.mean(sim.sinr_db)),
            "energy_j": sim.energy_j,
            "spectral_efficiency_bps_hz": sim.spectral_efficiency,
        }
        metadata = {
            **model_metadata,
            "pipeline": ["csi", "knn_graph", "gnn_gru" if torch is not None else "baseline"],
            "graph_nodes": int(len(graph.node_features)),
            "graph_edges": int(graph.edge_index.shape[1]),
            "torch_available": torch is not None,
            "refinement_invoked": refinement_invoked,
            "confidence_threshold": self.gate.min_confidence,
            "ood_threshold": self.gate.max_ood_score,
            "active_learning_buffer_size": len(self.buffer),
            "active_learning_reason": active_reason,
            "optimizer_iterations": int(optimizer_stats.get("iterations", 0)),
            "objective_evaluations": int(optimizer_stats.get("objective_evaluations", 0)),
            "runtime_ms": round((time.perf_counter() - started_at) * 1000.0, 3),
        }
        decision = {
            "controller_decision": "AI_ACCEPTED" if gate.accept else "CLASSICAL_REFINEMENT",
            "optimizer_called": refinement_invoked,
            "reason": "confidence >= threshold and OOD <= threshold" if gate.accept else gate.reason,
        }
        result = OptimizationResult(
            channels.tolist(), phases.tolist(), gate.accept, gate.reason,
            confidence, ood_score, sim.to_dict(), metrics, metadata, active_reason,
            uncertainty,
        )
        result.experiment_id = f"aurora-{int(started_at * 1000)}"
        result.scenario = {
            "num_users": int(state.config.num_users),
            "num_elements": int(state.config.num_elements),
            "num_channels": int(state.config.num_channels),
            "phase_resolution": int(state.config.phase_resolution),
            "environment": state.config.environment,
            "snr_db": float(state.config.snr_db),
            "csi_error": float(state.config.csi_error),
            "mobility": float(state.config.mobility),
            "seed": int(state.config.seed),
        }
        result.prediction = {
            "confidence": float(confidence),
            "ood_score": float(ood_score),
            "uncertainty": float(uncertainty),
        }
        result.gate = {
            "passed": bool(gate.accept),
            "confidence_threshold": float(self.gate.min_confidence),
            "ood_threshold": float(self.gate.max_ood_score),
            "reason": gate.reason,
        }
        result.decision = decision
        result.optimizer = {
            "called": bool(refinement_invoked),
            "iterations": int(optimizer_stats.get("iterations", 0)),
            "objective_evaluations": int(optimizer_stats.get("objective_evaluations", 0)),
            "runtime_ms": float(metadata["runtime_ms"]),
            "model": model_metadata.get("model", "deterministic-baseline"),
            "pipeline": metadata["pipeline"],
        }
        result.ris = {
            "elements": int(state.config.num_elements),
            "phase_resolution_bits": int(state.config.phase_resolution),
            "phases": [float(x) for x in phases.tolist()],
            "quantized": True,
        }
        result.comparison = {
            "before_refinement": {
                "throughput_mbps": float(pre_refinement_sim.throughput_mbps),
                "mean_sinr_db": float(np.mean(pre_refinement_sim.sinr_db)),
                "energy_j": float(pre_refinement_sim.energy_j),
                "spectral_efficiency_bps_hz": float(pre_refinement_sim.spectral_efficiency),
            },
            "after_refinement": {
                "throughput_mbps": float(sim.throughput_mbps),
                "mean_sinr_db": float(np.mean(sim.sinr_db)),
                "energy_j": float(sim.energy_j),
                "spectral_efficiency_bps_hz": float(sim.spectral_efficiency),
            },
        }
        return result

    def predict(self, scenario: Scenario | dict[str, Any] | None = None) -> OptimizationResult:
        """Compatibility alias for the public prediction endpoint."""
        return self.optimize(scenario)

    def _score(self, scenario: Scenario, channels: np.ndarray, phases: np.ndarray) -> float:
        return self.simulator.run(scenario, channels, phases).throughput_mbps
