"""Leakage-resistant NFL forecasting and betting research primitives.

The package is intentionally independent of the Streamlit UI.  It provides a
small, testable core that the legacy application can adopt incrementally.
"""

from .backtest import summarize_bets
from .contracts import GAME_FACT_CONTRACT, SchemaError, assert_point_in_time
from .features import build_pregame_features
from .markets import american_to_implied, de_vig_two_way, expected_profit
from .validation import season_walk_forward_splits

__all__ = [
    "GAME_FACT_CONTRACT",
    "SchemaError",
    "american_to_implied",
    "assert_point_in_time",
    "build_pregame_features",
    "de_vig_two_way",
    "expected_profit",
    "season_walk_forward_splits",
    "summarize_bets",
]

__version__ = "0.1.0"
