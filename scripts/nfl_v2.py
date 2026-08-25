"""Thin executable wrapper for :mod:`nfl_predictor.cli`."""

from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from nfl_predictor.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
