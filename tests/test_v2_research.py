import unittest

import pandas as pd

from nfl_predictor.research import benchmark_by_season, flat_shadow_decisions, require_promotable_shadow_period


class ResearchTests(unittest.TestCase):
    def test_benchmark_reports_model_and_market_without_slice_selection(self):
        frame = pd.DataFrame({
            "season": [2024, 2024, 2025, 2025],
            "target_home_win": [0, 1, 0, 1],
            "prob_home_win": [0.2, 0.8, 0.3, 0.7],
            "home_ml_implied_fair": [0.4, 0.6, 0.45, 0.55],
        })
        report = benchmark_by_season(frame)
        self.assertEqual(report["2024"]["targets"]["home_win"]["model"]["n"], 2)
        self.assertIn("brier", report["2025"]["targets"]["home_win"]["market"])

    def test_shadow_decisions_are_flat_and_require_exact_quote(self):
        candidates = pd.DataFrame({
            "prediction_id": ["p1", "p2"], "market_snapshot_id": ["m1", "m2"],
            "model_probability": [0.60, 0.51], "market_probability": [0.50, 0.50],
            "price_american": [100, -110],
        })
        decisions = flat_shadow_decisions(candidates, minimum_edge=0.025)
        self.assertEqual(decisions["decision"].tolist(), ["shadow", "pass"])
        self.assertTrue((decisions["stake_fraction"] == 0.0).all())

    def test_promotion_gate_fails_closed(self):
        with self.assertRaises(ValueError):
            require_promotable_shadow_period({"bets": 600, "roi": 0.02})


if __name__ == "__main__":
    unittest.main()
