import tempfile
import unittest
from pathlib import Path

import pandas as pd

from nfl_predictor.ingestion import ingest_file
from nfl_predictor.warehouse import connect, foreign_key_violations, initialize


class IngestionTests(unittest.TestCase):
    def test_file_ingestion_records_hash_and_normalized_game(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            games_path = root / "games.csv"
            pd.DataFrame([{
                "game_id": "2026_01_A_B", "season": 2026, "week": 1,
                "gameday": "2026-09-10", "gametime": "20:20", "home_team": "A", "away_team": "B",
            }]).to_csv(games_path, index=False)
            database = root / "v2.sqlite3"
            initialize(database)
            connection = connect(database)
            try:
                run_id, rows = ingest_file(
                    connection, path=games_path, source_name="fixture", source_uri="file://fixture",
                    available_at="2026-09-01T12:00:00Z", kind="games",
                )
                self.assertEqual(rows, 1)
                source = connection.execute("SELECT status, content_sha256 FROM source_run WHERE source_run_id = ?", (run_id,)).fetchone()
                game = connection.execute("SELECT kickoff_at, source_run_id FROM game").fetchone()
                self.assertEqual(source[0], "succeeded")
                self.assertEqual(len(source[1]), 64)
                self.assertTrue(game[0].endswith("Z"))
                self.assertEqual(game[1], run_id)
                self.assertTrue(foreign_key_violations(connection).empty)
            finally:
                connection.close()

    def test_invalid_market_file_is_recorded_as_failed_source_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "bad.csv"
            pd.DataFrame([{"game_id": "g1"}]).to_csv(path, index=False)
            database = root / "v2.sqlite3"
            initialize(database)
            connection = connect(database)
            try:
                with self.assertRaises(ValueError):
                    ingest_file(connection, path=path, source_name="fixture", source_uri=None, available_at="2026-01-01T00:00:00Z", kind="markets")
                self.assertEqual(connection.execute("SELECT status FROM source_run").fetchone()[0], "failed")
                self.assertEqual(connection.execute("SELECT rule_name FROM data_quality_event").fetchone()[0], "ingestion_failed")
            finally:
                connection.close()


if __name__ == "__main__":
    unittest.main()
