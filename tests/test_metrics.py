import unittest

from src.evaluation.metrics import (
    bit_error_rate,
    energy_consumption_j,
    packet_delivery_ratio,
    spectrum_efficiency_bps_per_hz,
    throughput_bps,
)


class MetricTests(unittest.TestCase):
    def test_wireless_metric_formulas(self) -> None:
        self.assertEqual(throughput_bps(10_000, 2), 5_000)
        self.assertEqual(packet_delivery_ratio(90, 100), 0.9)
        self.assertEqual(bit_error_rate(2, 1000), 0.002)
        self.assertEqual(energy_consumption_j(2.5, 4), 10)
        self.assertEqual(spectrum_efficiency_bps_per_hz(20_000_000, 20_000_000), 1)

    def test_invalid_metric_inputs_raise_errors(self) -> None:
        with self.assertRaises(ValueError):
            throughput_bps(10, 0)
        with self.assertRaises(ValueError):
            packet_delivery_ratio(101, 100)


if __name__ == "__main__":
    unittest.main()

