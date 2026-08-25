import unittest

import numpy as np
import pandas as pd

from nfl_predictor.features import add_game_targets, build_pregame_features, feature_columns


def sample_games() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["2024_01_B_A", "2024_02_B_A", "2024_03_B_A", "2024_04_B_A"],
            "season": [2024] * 4,
            "week": [1, 2, 3, 4],
            "gameday": ["2024-09-01", "2024-09-08", "2024-09-15", "2024-09-22"],
            "gametime": ["13:00", "13:00", "20:20", "16:25"],
            "home_team": ["A"] * 4,
            "away_team": ["B"] * 4,
            "home_score": [24, 14, 21, np.nan],
            "away_score": [17, 20, 21, np.nan],
            "spread_line": [3.5, -2.5, 0.0, 1.5],
            "total_line": [40.5, 42.5, 42.0, 44.5],
            "home_moneyline": [-170, 120, -110, -130],
            "away_moneyline": [150, -140, -110, 110],
            "home_spread_odds": [-110] * 4,
            "away_spread_odds": [-110] * 4,
            "over_odds": [-105] * 4,
            "under_odds": [-115] * 4,
            "home_rest": [7] * 4,
            "away_rest": [7] * 4,
            "roof": ["outdoors", "outdoors", "dome", "outdoors"],
            "temp": [60, 20, np.nan, 70],
            "wind": [18, 5, np.nan, 8],
            "div_game": [0, 1, 0, 0],
            "location": ["Home"] * 4,
        }
    )


class FeatureTests(unittest.TestCase):
    def test_nflverse_spread_convention(self):
        targets = add_game_targets(sample_games())
        self.assertEqual(targets.loc[0, "target_home_cover"], 1.0)
        self.assertEqual(targets.loc[1, "target_home_cover"], 0.0)
        self.assertTrue(pd.isna(targets.loc[2, "target_home_cover"]))

    def test_future_rows_do_not_become_losses(self):
        features = build_pregame_features(sample_games())
        self.assertFalse(features.loc[3, "is_completed"])
        self.assertTrue(pd.isna(features.loc[3, "target_home_win"]))
        self.assertAlmostEqual(features.loc[1, "home_points_for_mean_w3"], 24.0)
        self.assertAlmostEqual(features.loc[3, "home_points_for_mean_w3"], (24 + 14 + 21) / 3)

    def test_wind_flag_uses_wind_not_temperature(self):
        features = build_pregame_features(sample_games())
        self.assertEqual(features.loc[0, "is_windy"], 1)
        self.assertEqual(features.loc[1, "is_windy"], 0)
        self.assertEqual(features.loc[2, "is_windy"], 0)

    def test_legacy_close_cutoff_is_kickoff_not_midnight(self):
        features = build_pregame_features(sample_games())
        self.assertEqual(str(features.loc[0, "feature_cutoff_at"]), "2024-09-01 17:00:00+00:00")

    def test_pbp_windows_are_shifted_before_joining(self):
        rows = []
        for game_id, home_epa, away_epa in zip(
            sample_games()["game_id"], [0.4, 0.2, -0.1, 0.8], [-0.2, 0.1, 0.3, -0.4]
        ):
            rows.extend(
                [
                    {
                        "game_id": game_id,
                        "posteam": "A",
                        "defteam": "B",
                        "play_type": "pass",
                        "epa": home_epa,
                        "success": float(home_epa > 0),
                        "pass_attempt": 1,
                        "yards_gained": 8,
                        "down": 1,
                        "yardline_100": 50,
                        "wp": 0.5,
                    },
                    {
                        "game_id": game_id,
                        "posteam": "B",
                        "defteam": "A",
                        "play_type": "run",
                        "epa": away_epa,
                        "success": float(away_epa > 0),
                        "rush_attempt": 1,
                        "yards_gained": 5,
                        "down": 1,
                        "yardline_100": 50,
                        "wp": 0.5,
                    },
                ]
            )
        features = build_pregame_features(sample_games(), pbp=pd.DataFrame(rows))
        self.assertTrue(pd.isna(features.loc[0, "home_off_epa_per_play_mean_w4"]))
        self.assertAlmostEqual(features.loc[1, "home_off_epa_per_play_mean_w4"], 0.4)

    def test_feature_catalog_exceeds_thirty_and_excludes_outcomes(self):
        features = build_pregame_features(sample_games())
        columns = feature_columns(features)
        self.assertGreater(len(columns), 30)
        self.assertNotIn("home_score", columns)
        self.assertNotIn("target_home_win", columns)
        self.assertNotIn("actual_total", columns)
        self.assertNotIn("espn", columns)
        self.assertNotIn("old_game_id", columns)


if __name__ == "__main__":
    unittest.main()
