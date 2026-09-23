import unittest
import numpy as np

from backend.aurora.controller import AURORAController
from backend.aurora.gate import ConfidenceGate
from backend.aurora.graph import build_knn_graph
from backend.aurora.quantization import quantize_phases
from backend.aurora.scenario import Scenario, ScenarioConfig


class AuroraCoreTests(unittest.TestCase):
    def test_controller_is_deterministic(self):
        config = ScenarioConfig(seed=11, num_users=3, num_elements=4, num_channels=5)
        first = AURORAController(Scenario.generate(config), seed=11).optimize()
        second = AURORAController(Scenario.generate(config), seed=11).optimize()
        self.assertEqual(first.channels, second.channels)
        self.assertEqual(first.metrics, second.metrics)

    def test_confidence_gate_cases(self):
        gate = ConfidenceGate(min_confidence=0.75, max_ood_score=0.5)
        cases = [
            (0.90, 0.10, True, "AI_ACCEPTED"),
            (0.60, 0.70, False, "CLASSICAL_REFINEMENT"),
            (0.90, 0.60, False, "CLASSICAL_REFINEMENT"),
            (0.60, 0.20, False, "CLASSICAL_REFINEMENT"),
        ]
        for confidence, ood_score, accepted, expected_label in cases:
            decision = gate.decide(confidence, ood_score)
            self.assertEqual(decision.accept, accepted)
            self.assertEqual(("AI_ACCEPTED" if decision.accept else "CLASSICAL_REFINEMENT"), expected_label)

    def test_graph_and_phase_quantization(self):
        graph = build_knn_graph(np.eye(3), k=1)
        self.assertEqual(graph.edge_index.shape[0], 2)
        phases = quantize_phases(np.array([0.3, 2.0]), bits=2)
        self.assertTrue(np.all((phases >= 0) & (phases < 2 * np.pi)))


if __name__ == "__main__":
    unittest.main()
