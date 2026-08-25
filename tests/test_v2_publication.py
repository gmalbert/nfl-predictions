import json
import tempfile
import unittest
from pathlib import Path

from nfl_predictor.publication import publish_manifest


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


if __name__ == "__main__":
    unittest.main()
