import sqlite3
import unittest

from nfl_predictor.warehouse import SCHEMA_PATH, connect, foreign_key_violations


class WarehouseTests(unittest.TestCase):
    def test_schema_initializes_with_foreign_keys(self):
        with connect(":memory:") as connection:
            connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
            tables = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
            self.assertEqual(len(tables), 12)
            self.assertTrue(foreign_key_violations(connection).empty)

    def test_market_natural_key_rejects_duplicate_game_market(self):
        with connect(":memory:") as connection:
            connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
            connection.execute(
                """INSERT INTO game
                   (game_id, season, week, kickoff_at, home_team, away_team)
                   VALUES ('g1', 2026, 1, '2026-09-10T00:00:00Z', 'A', 'B')"""
            )
            values = (
                "m1",
                "g1",
                "book",
                "moneyline",
                "",
                "home",
                0.0,
                -110.0,
                "2026-09-09T12:00:00Z",
                "2026-09-09T12:00:01Z",
            )
            statement = """INSERT INTO market_snapshot
                (market_snapshot_id, game_id, book, market, participant_id, side,
                 line, price_american, observed_at, available_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
            connection.execute(statement, values)
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(statement, ("m2", *values[1:]))


if __name__ == "__main__":
    unittest.main()
