import unittest

from src.preprocessing.preprocess import (
    drop_rows_with_missing_values,
    validate_required_columns,
)


class PreprocessingTests(unittest.TestCase):
    def test_required_columns_are_validated(self) -> None:
        rows = [{"snr_db": 20, "interference_dbm": -90, "selected_channel": 6}]
        validate_required_columns(rows, {"snr_db", "interference_dbm", "selected_channel"})

    def test_missing_values_are_dropped(self) -> None:
        rows = [
            {"snr_db": 20, "selected_channel": 6},
            {"snr_db": "", "selected_channel": 11},
            {"snr_db": None, "selected_channel": 1},
        ]

        self.assertEqual(drop_rows_with_missing_values(rows), [{"snr_db": 20, "selected_channel": 6}])


if __name__ == "__main__":
    unittest.main()

