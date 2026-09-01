import unittest

import pandas as pd

from nfl_predictor.backtest import grade_bets, promotion_gate, summarize_bets


class BacktestTests(unittest.TestCase):
    def test_grading_and_summary(self):
        bets = pd.DataFrame(
            [
                {
                    "market": "moneyline",
                    "selection": "home",
                    "odds": -110,
                    "stake": 1.0,
                    "home_score": 24,
                    "away_score": 20,
                },
                {
                    "market": "spread",
                    "selection": "home",
                    "odds": -110,
                    "stake": 1.0,
                    "home_line": -3.0,
                    "home_score": 20,
                    "away_score": 17,
                },
                {
                    "market": "total",
                    "selection": "under",
                    "odds": -105,
                    "stake": 1.0,
                    "total_line": 44.5,
                    "home_score": 17,
                    "away_score": 14,
                },
            ]
        )
        graded = grade_bets(bets)
        self.assertEqual(graded["result"].tolist(), ["win", "push", "win"])
        summary = summarize_bets(graded)
        self.assertEqual(summary["bets"], 2)
        self.assertEqual(summary["pushes"], 1)
        self.assertGreater(summary["roi"], 0)

    def test_promotion_gate_is_conservative(self):
        promoted, reasons = promotion_gate({"bets": 20, "roi": 0.1, "mean_clv": 0.01})
        self.assertFalse(promoted)
        self.assertTrue(reasons)


if __name__ == "__main__":
    unittest.main()
