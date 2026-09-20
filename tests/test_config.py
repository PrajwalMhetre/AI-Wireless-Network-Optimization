import unittest

from src.utils.config import load_config


class ConfigTests(unittest.TestCase):
    def test_default_config_loads_required_sections(self) -> None:
        config = load_config()

        self.assertIn("simulation", config)
        self.assertIn("optimization", config)
        self.assertEqual(config["optimization"]["primary_decision"], "channel_selection")
        self.assertGreater(config["simulation"]["duration_s"], 0)


if __name__ == "__main__":
    unittest.main()

