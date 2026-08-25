import json
import tempfile
import unittest
from pathlib import Path

from nfl_predictor.publication import publish_manifest
from nfl_predictor.publication import record_shadow_decisions
from nfl_predictor.warehouse import connect, initialize
import pandas as pd


class PublicationTests(unittest.TestCase):
    def test_manifest_is_atomic_and_contains_reproducibility_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            result = publish_manifest(
                path, artifact_type="walk_forward", source_run_ids=["b", "a"],
                feature_set_version="v2", cutoff_at="2026-09-01T00:00:00Z", metrics={"brier": 0.2},
            )
            self.assertTrue(path.exists())
            self.assertFalse(path.with_suffix(".json.tmp").exists())
            self.assertEqual(result["source_run_ids"], ["a", "b"])
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["metrics"]["brier"], 0.2)

    def test_shadow_journal_requires_real_prediction_and_quote(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "v2.sqlite3"
            initialize(database)
            connection = connect(database)
            try:
                connection.execute("INSERT INTO game (game_id, season, week, kickoff_at, home_team, away_team) VALUES ('g', 2026, 1, '2026-09-10T00:00:00Z', 'A', 'B')")
                connection.execute("INSERT INTO model_run (model_run_id, model_name, model_version, feature_set_version, target_name, train_start_at, train_end_at, parameters_json, created_at) VALUES ('r', 'm', '1', 'f', 't', '2024-01-01', '2025-01-01', '{}', '2026-01-01')")
                connection.execute("INSERT INTO prediction_snapshot (prediction_id, model_run_id, game_id, market, participant_id, side, line, model_probability, cutoff_at, created_at) VALUES ('p', 'r', 'g', 'moneyline', '', 'home', 0, .6, '2026-09-01', '2026-09-01')")
                connection.execute("INSERT INTO market_snapshot (market_snapshot_id, game_id, book, market, participant_id, side, line, price_american, observed_at, available_at) VALUES ('q', 'g', 'book', 'moneyline', '', 'home', 0, 100, '2026-09-01', '2026-09-01')")
                rows = record_shadow_decisions(connection, pd.DataFrame([{"prediction_id": "p", "market_snapshot_id": "q", "decision": "shadow", "edge": .1, "expected_profit_per_unit": .2, "stake_fraction": 0.0}]), policy_version="shadow.v1")
                self.assertEqual(rows, 1)
                self.assertEqual(connection.execute("SELECT decision FROM bet_decision").fetchone()[0], "shadow")
            finally:
                connection.close()


if __name__ == "__main__":
    unittest.main()
