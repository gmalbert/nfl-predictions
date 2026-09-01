"""Bet grading, bankroll paths, and honest economic performance metrics."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .contracts import SchemaError, require_columns
from .markets import (
    expected_profit,
    kelly_fraction,
    profit_for_result,
    settle_moneyline,
    settle_spread,
    settle_total,
)


def recommend_bets(
    predictions: pd.DataFrame,
    *,
    minimum_edge: float = 0.025,
    kelly_multiplier: float = 0.25,
    bankroll_cap: float = 0.02,
) -> pd.DataFrame:
    """Filter a long-form price/prediction table into paper-bet decisions."""

    require_columns(
        predictions,
        ("game_id", "market", "selection", "model_probability", "fair_probability", "odds"),
        context="recommend_bets",
    )
    result = predictions.copy()
    result["edge"] = result["model_probability"] - result["fair_probability"]
    result["expected_profit_per_unit"] = result.apply(
        lambda row: expected_profit(row["model_probability"], row["odds"]), axis=1
    )
    result["stake_fraction"] = result.apply(
        lambda row: kelly_fraction(
            row["model_probability"],
            row["odds"],
            fraction=kelly_multiplier,
            cap=bankroll_cap,
        ),
        axis=1,
    )
    result = result[
        result["edge"].ge(minimum_edge)
        & result["expected_profit_per_unit"].gt(0)
        & result["stake_fraction"].gt(0)
    ]
    # Do not recommend mutually exclusive sides in the same game/market.
    result = result.sort_values(
        ["game_id", "market", "expected_profit_per_unit"],
        ascending=[True, True, False],
        kind="stable",
    ).drop_duplicates(["game_id", "market"], keep="first")
    return result.reset_index(drop=True)


def grade_bets(bets: pd.DataFrame, *, default_stake: float = 1.0) -> pd.DataFrame:
    """Grade moneyline, spread, and total bets with explicit push handling."""

    require_columns(
        bets,
        ("market", "selection", "odds", "home_score", "away_score"),
        context="grade_bets",
    )
    result = bets.copy()
    if "stake" not in result:
        result["stake"] = float(default_stake)

    def settle(row: pd.Series) -> str:
        market = str(row["market"]).lower()
        selection = str(row["selection"]).lower()
        if market == "moneyline":
            if selection not in {"home", "away"}:
                raise SchemaError("moneyline selection must be home or away")
            return settle_moneyline(
                selected_home=selection == "home",
                home_score=row["home_score"],
                away_score=row["away_score"],
            )
        if market == "spread":
            if selection not in {"home", "away"} or "home_line" not in row:
                raise SchemaError("spread rows require home/away selection and home_line")
            return settle_spread(
                selected_home=selection == "home",
                home_score=row["home_score"],
                away_score=row["away_score"],
                home_line=row["home_line"],
            )
        if market == "total":
            if selection not in {"over", "under"} or "total_line" not in row:
                raise SchemaError("total rows require over/under selection and total_line")
            return settle_total(
                selected_over=selection == "over",
                home_score=row["home_score"],
                away_score=row["away_score"],
                total_line=row["total_line"],
            )
        raise SchemaError(f"unsupported market: {market}")

    result["result"] = result.apply(settle, axis=1)
    result["profit"] = result.apply(
        lambda row: profit_for_result(row["result"], row["odds"], stake=row["stake"]),
        axis=1,
    )
    return result


def bankroll_path(bets: pd.DataFrame, *, starting_bankroll: float = 100.0) -> pd.DataFrame:
    require_columns(bets, ("profit",), context="bankroll_path")
    result = bets.copy()
    if "placed_at" in result:
        result = result.sort_values("placed_at", kind="stable")
    result["bankroll"] = float(starting_bankroll) + result["profit"].cumsum()
    result["peak_bankroll"] = result["bankroll"].cummax().clip(lower=starting_bankroll)
    result["drawdown"] = result["bankroll"] - result["peak_bankroll"]
    result["drawdown_pct"] = result["drawdown"] / result["peak_bankroll"]
    return result


def summarize_bets(bets: pd.DataFrame) -> dict[str, float | int]:
    require_columns(bets, ("result", "stake", "profit"), context="summarize_bets")
    settled = bets[bets["result"].isin(["win", "loss", "push"])].copy()
    risked = settled.loc[settled["result"].ne("push"), "stake"].sum()
    decisions = settled[settled["result"].isin(["win", "loss"])]
    path = bankroll_path(settled, starting_bankroll=100.0) if len(settled) else settled
    positive = settled.loc[settled["profit"].gt(0), "profit"].sum()
    negative = -settled.loc[settled["profit"].lt(0), "profit"].sum()
    returns = settled["profit"] / settled["stake"].replace(0, np.nan)
    summary: dict[str, float | int] = {
        "bets": int(len(decisions)),
        "wins": int(decisions["result"].eq("win").sum()),
        "losses": int(decisions["result"].eq("loss").sum()),
        "pushes": int(settled["result"].eq("push").sum()),
        "hit_rate": float(decisions["result"].eq("win").mean()) if len(decisions) else float("nan"),
        "profit": float(settled["profit"].sum()),
        "amount_risked": float(risked),
        "roi": float(settled["profit"].sum() / risked) if risked else float("nan"),
        "profit_factor": float(positive / negative) if negative else float("inf"),
        "max_drawdown": float(path["drawdown"].min()) if len(path) else 0.0,
        "max_drawdown_pct": float(path["drawdown_pct"].min()) if len(path) else 0.0,
        "return_std": float(returns.std(ddof=1)) if returns.notna().sum() > 1 else float("nan"),
    }
    if "clv" in settled:
        summary["mean_clv"] = float(pd.to_numeric(settled["clv"], errors="coerce").mean())
        summary["positive_clv_rate"] = float(
            pd.to_numeric(settled["clv"], errors="coerce").gt(0).mean()
        )
    return summary


def cluster_bootstrap_roi(
    bets: pd.DataFrame,
    *,
    cluster_column: str = "season",
    simulations: int = 2_000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float]:
    """Bootstrap whole seasons instead of pretending bets are independent."""

    require_columns(bets, (cluster_column, "stake", "profit"), context="cluster_bootstrap_roi")
    clusters = bets[cluster_column].dropna().unique()
    if len(clusters) < 2 or simulations < 100:
        raise ValueError("at least two clusters and 100 simulations are required")
    rng = np.random.default_rng(random_state)
    values: list[float] = []
    for _ in range(simulations):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        pieces = [bets[bets[cluster_column].eq(cluster)] for cluster in sampled]
        bootstrap = pd.concat(pieces, ignore_index=True)
        risked = bootstrap["stake"].sum()
        values.append(float(bootstrap["profit"].sum() / risked) if risked else np.nan)
    alpha = (1.0 - confidence) / 2.0
    return float(np.nanquantile(values, alpha)), float(np.nanquantile(values, 1.0 - alpha))


def promotion_gate(
    summary: dict[str, float | int],
    *,
    minimum_bets: int = 500,
    require_positive_clv: bool = True,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if int(summary.get("bets", 0)) < minimum_bets:
        reasons.append(f"requires at least {minimum_bets} settled bets")
    if float(summary.get("roi", float("nan"))) <= 0:
        reasons.append("out-of-sample ROI is not positive")
    if require_positive_clv and float(summary.get("mean_clv", float("nan"))) <= 0:
        reasons.append("mean closing-line value is not positive")
    return not reasons, reasons
