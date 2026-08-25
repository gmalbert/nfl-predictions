import unittest

import pandas as pd

from nfl_predictor.contracts import SchemaError, TableContract, assert_point_in_time


class ContractTests(unittest.TestCase):
    def test_duplicate_key_is_rejected(self):
        contract = TableContract("demo", required=("id",), key=("id",))
        with self.assertRaises(SchemaError):
            contract.validate(pd.DataFrame({"id": [1, 1]}))

    def test_point_in_time_rejects_late_observation(self):
        frame = pd.DataFrame(
            {
                "game_id": ["g1"],
                "cutoff_at": ["2026-09-01T17:00:00Z"],
                "available_at": ["2026-09-01T17:01:00Z"],
            }
        )
        with self.assertRaises(SchemaError):
            assert_point_in_time(frame)

    def test_point_in_time_accepts_available_observation(self):
        frame = pd.DataFrame(
            {
                "cutoff_at": ["2026-09-01T17:00:00Z"],
                "available_at": ["2026-09-01T16:59:00Z"],
            }
        )
        assert_point_in_time(frame)


if __name__ == "__main__":
    unittest.main()
