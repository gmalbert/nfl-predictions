"""Calibrated probability and coherent score-distribution models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss

from .contracts import SchemaError


class IsotonicProbabilityCalibrator:
    """Calibration fitted only on an explicitly supplied holdout set."""

    def __init__(self) -> None:
        self._model: IsotonicRegression | None = None

    def fit(self, probability: Iterable[float], target: Iterable[float]) -> "IsotonicProbabilityCalibrator":
        x = np.asarray(list(probability), dtype=float)
        y = np.asarray(list(target), dtype=float)
        if x.shape != y.shape or len(x) < 20 or len(np.unique(y)) < 2:
            raise ValueError("isotonic calibration requires 20+ observations and both classes")
        if ((x < 0) | (x > 1)).any() or not np.isfinite(x).all():
            raise ValueError("calibration probabilities must be finite and in [0, 1]")
        self._model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._model.fit(x, y)
        return self

    def predict(self, probability: Iterable[float]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("calibrator has not been fitted")
        values = np.asarray(list(probability), dtype=float)
        return np.asarray(self._model.predict(values), dtype=float)


def learn_market_blend_weight(
    model_probability: Iterable[float],
    market_probability: Iterable[float],
    target: Iterable[float],
    *,
    grid: Sequence[float] | None = None,
) -> float:
    """Learn a convex model/market blend on calibration data only."""

    model = np.asarray(list(model_probability), dtype=float)
    market = np.asarray(list(market_probability), dtype=float)
    y = np.asarray(list(target), dtype=float)
    if model.shape != market.shape or model.shape != y.shape:
        raise ValueError("model, market, and target arrays must align")
    candidates = np.asarray(grid if grid is not None else np.linspace(0, 1, 101), dtype=float)
    losses = []
    for model_weight in candidates:
        blended = np.clip(model_weight * model + (1 - model_weight) * market, 1e-9, 1 - 1e-9)
        losses.append(log_loss(y, blended, labels=[0, 1]))
    return float(candidates[int(np.argmin(losses))])


def blend_with_market(
    model_probability: Iterable[float], market_probability: Iterable[float], *, model_weight: float
) -> np.ndarray:
    if not 0 <= model_weight <= 1:
        raise ValueError("model_weight must be in [0, 1]")
    model = np.asarray(list(model_probability), dtype=float)
    market = np.asarray(list(market_probability), dtype=float)
    if model.shape != market.shape:
        raise ValueError("probability arrays must align")
    return np.clip(model_weight * model + (1 - model_weight) * market, 0.0, 1.0)


@dataclass
class ScoreSimulation:
    expected_margin: np.ndarray
    expected_total: np.ndarray
    margin_std: np.ndarray
    total_std: np.ndarray
    home_win_probability: np.ndarray
    home_cover_probability: np.ndarray
    over_probability: np.ndarray
    margin_p10: np.ndarray
    margin_p90: np.ndarray
    total_p10: np.ndarray
    total_p90: np.ndarray

    def to_frame(self, index: pd.Index | None = None) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "predicted_margin": self.expected_margin,
                "predicted_total": self.expected_total,
                "predicted_margin_std": self.margin_std,
                "predicted_total_std": self.total_std,
                "prob_home_win": self.home_win_probability,
                "prob_home_cover": self.home_cover_probability,
                "prob_over": self.over_probability,
                "margin_p10": self.margin_p10,
                "margin_p90": self.margin_p90,
                "total_p10": self.total_p10,
                "total_p90": self.total_p90,
            },
            index=index,
        )


class ScoreDistributionForecaster:
    """Predict margin and total, then simulate their correlated uncertainty."""

    def __init__(
        self,
        *,
        learning_rate: float = 0.05,
        max_iter: int = 300,
        max_leaf_nodes: int = 15,
        l2_regularization: float = 1.0,
        random_state: int = 42,
    ) -> None:
        kwargs = dict(
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
            l2_regularization=l2_regularization,
            random_state=random_state,
        )
        self.margin_model = HistGradientBoostingRegressor(loss="squared_error", **kwargs)
        self.total_model = HistGradientBoostingRegressor(loss="squared_error", **kwargs)
        self.random_state = random_state
        self.feature_names_: list[str] = []
        self.residual_covariance_: np.ndarray | None = None

    def fit(
        self,
        features: pd.DataFrame,
        home_margin: Iterable[float],
        actual_total: Iterable[float],
    ) -> "ScoreDistributionForecaster":
        x = self._numeric_frame(features, fitting=True)
        margin = np.asarray(list(home_margin), dtype=float)
        total = np.asarray(list(actual_total), dtype=float)
        if len(x) != len(margin) or len(x) != len(total):
            raise ValueError("features and targets must have the same number of rows")
        valid = np.isfinite(margin) & np.isfinite(total)
        if valid.sum() < 100:
            raise ValueError("score model requires at least 100 completed games")
        self.margin_model.fit(x.loc[valid], margin[valid])
        self.total_model.fit(x.loc[valid], total[valid])
        residuals = np.column_stack(
            [
                margin[valid] - self.margin_model.predict(x.loc[valid]),
                total[valid] - self.total_model.predict(x.loc[valid]),
            ]
        )
        covariance = np.cov(residuals, rowvar=False)
        if covariance.shape != (2, 2) or not np.isfinite(covariance).all():
            raise SchemaError("could not estimate score residual covariance")
        covariance += np.eye(2) * 1e-6
        self.residual_covariance_ = covariance
        return self

    def predict_distribution(
        self,
        features: pd.DataFrame,
        *,
        spread_line: Iterable[float],
        total_line: Iterable[float],
        simulations: int = 10_000,
        random_state: int | None = None,
    ) -> ScoreSimulation:
        if self.residual_covariance_ is None:
            raise RuntimeError("score model has not been fitted")
        if simulations < 1_000:
            raise ValueError("use at least 1,000 simulations for stable probabilities")
        x = self._numeric_frame(features, fitting=False)
        spread = np.asarray(list(spread_line), dtype=float)
        total_threshold = np.asarray(list(total_line), dtype=float)
        if len(x) != len(spread) or len(x) != len(total_threshold):
            raise ValueError("features and market lines must align")
        margin_center = self.margin_model.predict(x)
        total_center = self.total_model.predict(x)
        rng = np.random.default_rng(self.random_state if random_state is None else random_state)
        noise = rng.multivariate_normal(
            mean=[0.0, 0.0], cov=self.residual_covariance_, size=(len(x), simulations)
        )
        margin_draws = margin_center[:, None] + noise[:, :, 0]
        total_draws = total_center[:, None] + noise[:, :, 1]
        return ScoreSimulation(
            expected_margin=margin_center,
            expected_total=total_center,
            margin_std=margin_draws.std(axis=1, ddof=1),
            total_std=total_draws.std(axis=1, ddof=1),
            home_win_probability=(margin_draws > 0).mean(axis=1),
            # nflverse spread_line is the points by which home is favored.
            home_cover_probability=(margin_draws > spread[:, None]).mean(axis=1),
            over_probability=(total_draws > total_threshold[:, None]).mean(axis=1),
            margin_p10=np.quantile(margin_draws, 0.10, axis=1),
            margin_p90=np.quantile(margin_draws, 0.90, axis=1),
            total_p10=np.quantile(total_draws, 0.10, axis=1),
            total_p90=np.quantile(total_draws, 0.90, axis=1),
        )

    def _numeric_frame(self, frame: pd.DataFrame, *, fitting: bool) -> pd.DataFrame:
        if fitting:
            numeric = frame.select_dtypes(include=["number", "bool"]).copy()
            if numeric.empty:
                raise ValueError("score model requires numeric features")
            self.feature_names_ = numeric.columns.tolist()
            return numeric.astype(float)
        missing = sorted(set(self.feature_names_) - set(frame.columns))
        if missing:
            raise SchemaError(f"inference frame is missing model features: {missing}")
        return frame[self.feature_names_].astype(float)


class EloRatings:
    """Leakage-free, margin-aware Elo baseline for game-level benchmarking."""

    def __init__(self, *, initial: float = 1500.0, k_factor: float = 20.0, home_advantage: float = 35.0):
        self.initial = float(initial)
        self.k_factor = float(k_factor)
        self.home_advantage = float(home_advantage)
        self.ratings: dict[str, float] = {}

    def rating(self, team: str) -> float:
        return self.ratings.get(team, self.initial)

    def probability(self, home_team: str, away_team: str, *, neutral: bool = False) -> float:
        advantage = 0.0 if neutral else self.home_advantage
        difference = self.rating(home_team) + advantage - self.rating(away_team)
        return float(1.0 / (1.0 + 10.0 ** (-difference / 400.0)))

    def update(self, home_team: str, away_team: str, home_score: float, away_score: float) -> None:
        probability = self.probability(home_team, away_team)
        outcome = 1.0 if home_score > away_score else 0.0 if home_score < away_score else 0.5
        margin = abs(float(home_score) - float(away_score))
        multiplier = np.log1p(margin) * (2.2 / (2.2 + abs(self.rating(home_team) - self.rating(away_team)) * 0.001))
        change = self.k_factor * multiplier * (outcome - probability)
        self.ratings[home_team] = self.rating(home_team) + change
        self.ratings[away_team] = self.rating(away_team) - change

    def transform_games(self, games: pd.DataFrame) -> pd.DataFrame:
        required = {"gameday", "game_id", "home_team", "away_team", "home_score", "away_score"}
        missing = sorted(required - set(games.columns))
        if missing:
            raise SchemaError(f"Elo transform missing columns: {missing}")
        rows = []
        for row in games.sort_values(["gameday", "game_id"], kind="stable").itertuples():
            home_rating = self.rating(row.home_team)
            away_rating = self.rating(row.away_team)
            probability = self.probability(
                row.home_team,
                row.away_team,
                neutral=str(getattr(row, "location", "Home")).lower() != "home",
            )
            rows.append(
                {
                    "game_id": row.game_id,
                    "home_elo_pre": home_rating,
                    "away_elo_pre": away_rating,
                    "elo_diff_pre": home_rating - away_rating,
                    "elo_home_win_probability": probability,
                }
            )
            if pd.notna(row.home_score) and pd.notna(row.away_score):
                self.update(row.home_team, row.away_team, row.home_score, row.away_score)
        return pd.DataFrame(rows)
