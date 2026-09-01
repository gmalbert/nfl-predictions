"""Predeclared benchmark reporting and flat-stake shadow-decision helpers."""

from __future__ import annotations

from typing import Mapping

import pandas as pd

from .contracts import SchemaError, require_columns
from .markets import expected_profit
from .validation import probability_metrics


DEFAULT_TARGETS: Mapping[str, tuple[str, str, str | None]] = {
    "home_win": ("target_home_win", "prob_home_win", "home_ml_implied_fair"),
    "home_cover": ("target_home_cover", "prob_home_cover", None),
    "over": ("target_over", "prob_over", "over_implied_fair"),
}


def benchmark_report(frame: pd.DataFrame, *, targets: Mapping[str, tuple[str, str, str | None]] = DEFAULT_TARGETS) -> dict[str, object]:
    """Report model and available market baselines without selecting a slice.

    A missing price-derived probability is reported as unavailable rather than
    quietly substituted with a model result.
    """

    report: dict[str, object] = {"rows": int(len(frame)), "targets": {}}
    for name, (target_column, model_column, market_column) in targets.items():
        if target_column not in frame or model_column not in frame:
            continue
        valid = frame[[target_column, model_column]].dropna()
        if valid.empty:
            continue
        target_report: dict[str, object] = {
            "model": probability_metrics(valid[target_column], valid[model_column]),
        }
        if market_column and market_column in frame:
            market = frame[[target_column, market_column]].dropna()
            target_report["market"] = (
                probability_metrics(market[target_column], market[market_column])
                if not market.empty else {"status": "unavailable"}
            )
        else:
            target_report["market"] = {"status": "unavailable"}
        report["targets"][name] = target_report
    return report


def benchmark_by_season(frame: pd.DataFrame) -> dict[str, object]:
    require_columns(frame, ("season",), context="benchmark_by_season")
    return {
        str(int(season)): benchmark_report(group)
        for season, group in frame.groupby("season", sort=True)
    }


def flat_shadow_decisions(
    candidates: pd.DataFrame,
    *,
    minimum_edge: float = 0.025,
    paper_stake: float = 1.0,
) -> pd.DataFrame:
    """Create an auditable, no-Kelly shadow ledger from exact quoted prices."""

    require_columns(
        candidates,
        ("prediction_id", "market_snapshot_id", "model_probability", "market_probability", "price_american"),
        context="flat_shadow_decisions",
    )
    if paper_stake <= 0:
        raise ValueError("paper_stake must be positive")
    result = candidates.copy()
    result["edge"] = result["model_probability"] - result["market_probability"]
    result["expected_profit_per_unit"] = result.apply(
        lambda row: expected_profit(row["model_probability"], row["price_american"]), axis=1
    )
    result["decision"] = "pass"
    eligible = result["edge"].ge(minimum_edge) & result["expected_profit_per_unit"].gt(0)
    result.loc[eligible, "decision"] = "shadow"
    result["stake_fraction"] = 0.0
    result["paper_stake"] = float(paper_stake)
    return result


def require_promotable_shadow_period(summary: dict[str, object], *, minimum_bets: int = 500) -> None:
    """Fail closed unless the declared research gate has concrete evidence."""

    missing = [key for key in ("bets", "roi", "mean_clv", "roi_ci_low") if key not in summary]
    if missing:
        raise SchemaError(f"promotion evidence missing fields: {missing}")
    if int(summary["bets"]) < minimum_bets or float(summary["roi"]) <= 0 or float(summary["mean_clv"]) <= 0 or float(summary["roi_ci_low"]) <= 0:
        raise SchemaError("shadow period does not pass the predeclared promotion gate")
