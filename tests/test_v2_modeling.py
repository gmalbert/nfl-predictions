import unittest

import numpy as np
import pandas as pd

from nfl_predictor.modeling import EloRatings, ScoreDistributionForecaster


class ModelingTests(unittest.TestCase):
    def test_elo_features_are_pregame(self):
        games = pd.DataFrame(
            {
                "game_id": ["g1", "g2"],
                "gameday": ["2024-09-01", "2024-09-08"],
                "home_team": ["A", "A"],
                "away_team": ["B", "B"],
                "home_score": [30, np.nan],
                "away_score": [10, np.nan],
                "location": ["Home", "Home"],
            }
        )
        transformed = EloRatings().transform_games(games)
        self.assertEqual(transformed.loc[0, "home_elo_pre"], 1500.0)
        self.assertGreater(transformed.loc[1, "home_elo_pre"], 1500.0)

    def test_score_distribution_outputs_probabilities(self):
        rng = np.random.default_rng(4)
        rows = 140
        features = pd.DataFrame({"strength": rng.normal(size=rows), "market": rng.normal(size=rows)})
        margin = 4 * features["strength"] + rng.normal(0, 7, rows)
        total = 44 + 2 * features["market"] + rng.normal(0, 8, rows)
        model = ScoreDistributionForecaster(max_iter=40).fit(features, margin, total)
        simulation = model.predict_distribution(
            features.head(3),
            spread_line=[3.0, -2.0, 0.0],
            total_line=[44.5, 41.5, 48.0],
            simulations=1_000,
        )
        self.assertTrue(((simulation.home_win_probability >= 0) & (simulation.home_win_probability <= 1)).all())
        self.assertEqual(len(simulation.expected_margin), 3)

    def test_score_model_drops_all_missing_source_gated_features(self):
        rng = np.random.default_rng(7)
        rows = 120
        features = pd.DataFrame({"signal": rng.normal(size=rows), "unavailable_source": np.nan})
        model = ScoreDistributionForecaster(max_iter=20).fit(
            features, 3 * features["signal"] + rng.normal(size=rows), 44 + rng.normal(size=rows)
        )
        self.assertNotIn("unavailable_source", model.feature_names_)


if __name__ == "__main__":
    unittest.main()
