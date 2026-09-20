import unittest
from pathlib import Path


class ProjectStructureTests(unittest.TestCase):
    def test_phase_one_directories_exist(self) -> None:
        root = Path(__file__).resolve().parents[1]
        expected_dirs = [
            "config",
            "simulation/ns3/scenarios",
            "simulation/ns3/models",
            "simulation/ns3/scripts",
            "data/raw",
            "data/processed",
            "data/datasets",
            "results/figures",
            "results/metrics",
            "results/reports",
            "docs",
            "src",
            "tests",
        ]

        for relative_path in expected_dirs:
            self.assertTrue((root / relative_path).is_dir(), relative_path)


if __name__ == "__main__":
    unittest.main()

