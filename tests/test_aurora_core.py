import unittest
import numpy as np

from backend.aurora.controller import AURORAController
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

    def test_graph_and_phase_quantization(self):
        graph = build_knn_graph(np.eye(3), k=1)
        self.assertEqual(graph.edge_index.shape[0], 2)
        phases = quantize_phases(np.array([0.3, 2.0]), bits=2)
        self.assertTrue(np.all((phases >= 0) & (phases < 2 * np.pi)))


if __name__ == "__main__":
    unittest.main()
