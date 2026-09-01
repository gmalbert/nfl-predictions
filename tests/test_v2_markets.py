import unittest

from nfl_predictor.markets import (
    Price,
    american_to_decimal,
    american_to_implied,
    best_price,
    de_vig_two_way,
    expected_profit,
    kelly_fraction,
    settle_spread,
    settle_total,
)


class MarketMathTests(unittest.TestCase):
    def test_american_odds_conversion(self):
        self.assertAlmostEqual(american_to_decimal(-110), 1.9090909)
        self.assertAlmostEqual(american_to_decimal(150), 2.5)
        self.assertAlmostEqual(american_to_implied(-110), 0.5238095)

    def test_devig_sums_to_one(self):
        side_a, side_b = de_vig_two_way(-110, -110)
        self.assertAlmostEqual(side_a + side_b, 1.0)
        self.assertAlmostEqual(side_a, 0.5)

    def test_expected_value_and_fractional_kelly(self):
        self.assertGreater(expected_profit(0.56, -110), 0)
        self.assertEqual(kelly_fraction(0.50, -110), 0)
        self.assertLessEqual(kelly_fraction(0.60, -110, cap=0.02), 0.02)

    def test_spread_and_total_pushes(self):
        self.assertEqual(
            settle_spread(selected_home=True, home_score=24, away_score=21, home_line=-3),
            "push",
        )
        self.assertEqual(
            settle_total(selected_over=True, home_score=24, away_score=21, total_line=45),
            "push",
        )

    def test_best_price_prefers_higher_payout(self):
        selected = best_price([Price("a", -115), Price("b", -105)])
        self.assertEqual(selected.book, "b")


if __name__ == "__main__":
    unittest.main()
