import unittest

import numpy as np
import pandas as pd

from nfl_predictor.validation import (
    chronological_holdout,
    expected_calibration_error,
    probability_metrics,
    season_walk_forward_splits,
)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        rows = []
        for season in range(2019, 2025):
            for week in range(1, 4):
                rows.append(
                    {
                        "season": season,
                        "week": week,
                        "gameday": f"{season}-09-{week:02d}",
                    }
                )
        self.frame = pd.DataFrame(rows)

    def test_walk_forward_is_strictly_ordered(self):
        folds = season_walk_forward_splits(self.frame, first_test_season=2022)
        self.assertEqual([fold.test_season for fold in folds], [2022, 2023, 2024])
        first = folds[0]
        self.assertLess(
            self.frame.loc[first.train_index, "season"].max(),
            self.frame.loc[first.calibration_index, "season"].min(),
        )
        self.assertLess(
            self.frame.loc[first.calibration_index, "season"].max(),
            self.frame.loc[first.test_index, "season"].min(),
        )

    def test_chronological_holdout_does_not_shuffle(self):
        train, validation, test = chronological_holdout(self.frame)
        self.assertLess(max(train), min(validation))
        self.assertLess(max(validation), min(test))

    def test_probability_metrics(self):
        target = np.array([0, 0, 1, 1])
        probability = np.array([0.1, 0.2, 0.8, 0.9])
        metrics = probability_metrics(target, probability)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertLess(metrics["brier"], 0.05)
        self.assertGreaterEqual(expected_calibration_error(target, probability), 0)


if __name__ == "__main__":
    unittest.main()
