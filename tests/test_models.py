import unittest

from src.models.baseline import ChannelState, LowestInterferenceChannelSelector


class BaselineModelTests(unittest.TestCase):
    def test_lowest_interference_selector_is_deterministic(self) -> None:
        selector = LowestInterferenceChannelSelector()
        channels = [
            ChannelState(channel_id=1, interference_dbm=-80, snr_db=20),
            ChannelState(channel_id=6, interference_dbm=-90, snr_db=17),
            ChannelState(channel_id=11, interference_dbm=-90, snr_db=22),
        ]

        self.assertEqual(selector.select_channel(channels), 11)


if __name__ == "__main__":
    unittest.main()

