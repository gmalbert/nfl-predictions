"""Odds normalization, fair probabilities, expected value, and settlement."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable

import numpy as np


def _finite(value: float, *, name: str) -> float:
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def american_to_decimal(odds: float) -> float:
    odds = _finite(odds, name="odds")
    if odds == 0:
        raise ValueError("American odds cannot be zero")
    return 1.0 + (odds / 100.0 if odds > 0 else 100.0 / abs(odds))


def american_to_implied(odds: float) -> float:
    """Return the break-even probability including bookmaker margin."""

    return 1.0 / american_to_decimal(odds)


def decimal_to_american(decimal_odds: float) -> float:
    decimal_odds = _finite(decimal_odds, name="decimal_odds")
    if decimal_odds <= 1:
        raise ValueError("Decimal odds must be greater than one")
    return (decimal_odds - 1.0) * 100.0 if decimal_odds >= 2 else -100.0 / (decimal_odds - 1.0)


def de_vig(probabilities: Iterable[float], *, method: str = "multiplicative") -> np.ndarray:
    """Remove overround from a complete market.

    ``multiplicative`` normalizes probabilities by their sum. ``additive``
    subtracts an equal share of overround and then renormalizes.  The latter is
    useful as a sensitivity check, not as a universal truth model.
    """

    probs = np.asarray(list(probabilities), dtype=float)
    if probs.ndim != 1 or len(probs) < 2 or not np.isfinite(probs).all():
        raise ValueError("probabilities must contain at least two finite values")
    if (probs <= 0).any() or (probs >= 1).any():
        raise ValueError("raw implied probabilities must be between zero and one")
    overround = probs.sum() - 1.0
    if method == "multiplicative":
        fair = probs / probs.sum()
    elif method == "additive":
        fair = probs - overround / len(probs)
        if (fair <= 0).any():
            raise ValueError("additive de-vig produced a non-positive probability")
        fair = fair / fair.sum()
    else:
        raise ValueError("method must be 'multiplicative' or 'additive'")
    return fair


def de_vig_two_way(odds_a: float, odds_b: float, *, method: str = "multiplicative") -> tuple[float, float]:
    fair = de_vig(
        [american_to_implied(odds_a), american_to_implied(odds_b)],
        method=method,
    )
    return float(fair[0]), float(fair[1])


def expected_profit(probability: float, american_odds: float, *, stake: float = 1.0) -> float:
    probability = _finite(probability, name="probability")
    stake = _finite(stake, name="stake")
    if not 0 <= probability <= 1 or stake < 0:
        raise ValueError("probability must be in [0, 1] and stake must be non-negative")
    net_win = stake * (american_to_decimal(american_odds) - 1.0)
    return probability * net_win - (1.0 - probability) * stake


def kelly_fraction(
    probability: float,
    american_odds: float,
    *,
    fraction: float = 0.25,
    cap: float = 0.02,
) -> float:
    """Fractional Kelly with a hard bankroll cap and no-bet floor at zero."""

    probability = _finite(probability, name="probability")
    fraction = _finite(fraction, name="fraction")
    cap = _finite(cap, name="cap")
    if not 0 <= probability <= 1 or not 0 <= fraction <= 1 or not 0 <= cap <= 1:
        raise ValueError("probability, fraction, and cap must be bounded probabilities")
    b = american_to_decimal(american_odds) - 1.0
    full_kelly = (b * probability - (1.0 - probability)) / b
    return float(min(cap, max(0.0, full_kelly * fraction)))


def settle_moneyline(*, selected_home: bool, home_score: float, away_score: float) -> str:
    if home_score == away_score:
        return "push"
    won = home_score > away_score if selected_home else away_score > home_score
    return "win" if won else "loss"


def settle_spread(
    *,
    selected_home: bool,
    home_score: float,
    away_score: float,
    home_line: float,
) -> str:
    """Settle a spread using the selected side's signed handicap.

    Example: a home favorite listed at -3.5 has ``home_line=-3.5``.
    """

    adjusted = (home_score - away_score) + home_line
    selected_margin = adjusted if selected_home else -adjusted
    if np.isclose(selected_margin, 0.0):
        return "push"
    return "win" if selected_margin > 0 else "loss"


def settle_total(*, selected_over: bool, home_score: float, away_score: float, total_line: float) -> str:
    difference = (home_score + away_score) - total_line
    selected_margin = difference if selected_over else -difference
    if np.isclose(selected_margin, 0.0):
        return "push"
    return "win" if selected_margin > 0 else "loss"


def profit_for_result(result: str, american_odds: float, *, stake: float = 1.0) -> float:
    if result == "push":
        return 0.0
    if result == "loss":
        return -float(stake)
    if result != "win":
        raise ValueError("result must be win, loss, or push")
    return float(stake) * (american_to_decimal(american_odds) - 1.0)


def probability_clv(*, bet_probability: float, close_probability: float) -> float:
    """Closing-line value in no-vig probability points."""

    return _finite(close_probability, name="close_probability") - _finite(
        bet_probability, name="bet_probability"
    )


@dataclass(frozen=True)
class Price:
    book: str
    american_odds: float
    line: float | None = None


def best_price(prices: Iterable[Price]) -> Price:
    """Choose the highest decimal payout, breaking ties toward the better line."""

    candidates = list(prices)
    if not candidates:
        raise ValueError("at least one price is required")
    return max(
        candidates,
        key=lambda price: (
            american_to_decimal(price.american_odds),
            float("-inf") if price.line is None else price.line,
        ),
    )
