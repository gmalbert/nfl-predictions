"""Temporal evaluation, probability metrics, and leakage diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from .contracts import SchemaError, require_columns


@dataclass(frozen=True)
class WalkForwardFold:
    test_season: int
    train_index: np.ndarray
    calibration_index: np.ndarray
    test_index: np.ndarray

    @property
    def label(self) -> str:
        return f"season_{self.test_season}"


def season_walk_forward_splits(
    frame: pd.DataFrame,
    *,
    first_test_season: int | None = None,
    calibration_seasons: int = 1,
    minimum_train_seasons: int = 2,
    embargo_days: int = 0,
    season_column: str = "season",
    time_column: str = "gameday",
) -> list[WalkForwardFold]:
    """Create expanding, season-forward train/calibration/test folds.

    Thresholds and probability calibration belong in ``calibration_index``;
    final reporting belongs in ``test_index``.  No target-season row is exposed
    to fitting or tuning.
    """

    require_columns(frame, (season_column, time_column), context="season_walk_forward_splits")
    if calibration_seasons < 1 or minimum_train_seasons < 1 or embargo_days < 0:
        raise ValueError("season counts must be positive and embargo_days non-negative")
    seasons = pd.to_numeric(frame[season_column], errors="raise").astype(int)
    timestamps = pd.to_datetime(frame[time_column], utc=True, errors="raise")
    unique = sorted(seasons.dropna().unique().tolist())
    earliest_position = minimum_train_seasons + calibration_seasons
    if first_test_season is not None:
        unique = [season for season in unique if season >= int(first_test_season)]
    else:
        unique = unique[earliest_position:]

    all_seasons = sorted(seasons.dropna().unique().tolist())
    folds: list[WalkForwardFold] = []
    for test_season in unique:
        prior = [season for season in all_seasons if season < test_season]
        if len(prior) < minimum_train_seasons + calibration_seasons:
            continue
        calibration_values = prior[-calibration_seasons:]
        train_values = prior[:-calibration_seasons]
        train_mask = seasons.isin(train_values)
        calibration_mask = seasons.isin(calibration_values)
        test_mask = seasons.eq(test_season)
        if embargo_days:
            test_start = timestamps.loc[test_mask].min()
            train_mask &= timestamps < test_start - pd.Timedelta(days=embargo_days)

        fold = WalkForwardFold(
            test_season=int(test_season),
            train_index=frame.index[train_mask].to_numpy(),
            calibration_index=frame.index[calibration_mask].to_numpy(),
            test_index=frame.index[test_mask].to_numpy(),
        )
        _validate_fold(frame, fold, time_column=time_column)
        folds.append(fold)
    if not folds:
        raise SchemaError("not enough seasons to create a walk-forward fold")
    return folds


def _validate_fold(frame: pd.DataFrame, fold: WalkForwardFold, *, time_column: str) -> None:
    train = set(fold.train_index)
    calibration = set(fold.calibration_index)
    test = set(fold.test_index)
    if not train or not calibration or not test:
        raise SchemaError(f"{fold.label}: train, calibration, and test must be non-empty")
    if train & calibration or train & test or calibration & test:
        raise SchemaError(f"{fold.label}: fold indices overlap")
    timestamps = pd.to_datetime(frame[time_column], utc=True)
    if timestamps.loc[list(train)].max() >= timestamps.loc[list(calibration)].min():
        raise SchemaError(f"{fold.label}: training does not precede calibration")
    if timestamps.loc[list(calibration)].max() >= timestamps.loc[list(test)].min():
        raise SchemaError(f"{fold.label}: calibration does not precede test")


def chronological_holdout(
    frame: pd.DataFrame,
    *,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    time_column: str = "gameday",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    require_columns(frame, (time_column,), context="chronological_holdout")
    if validation_fraction <= 0 or test_fraction <= 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("validation and test fractions must be positive and sum to less than one")
    order = frame.assign(
        __time=pd.to_datetime(frame[time_column], utc=True, errors="raise")
    ).sort_values(["__time"], kind="stable").index.to_numpy()
    n = len(order)
    train_end = int(n * (1 - validation_fraction - test_fraction))
    validation_end = int(n * (1 - test_fraction))
    if train_end < 1 or validation_end <= train_end or validation_end >= n:
        raise SchemaError("frame is too small for chronological train/validation/test split")
    return order[:train_end], order[train_end:validation_end], order[validation_end:]


def expected_calibration_error(
    y_true: Iterable[float],
    y_probability: Iterable[float],
    *,
    bins: int = 10,
) -> float:
    y = np.asarray(list(y_true), dtype=float)
    probability = np.asarray(list(y_probability), dtype=float)
    _validate_probabilities(y, probability)
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_id = np.clip(np.digitize(probability, edges[1:-1], right=False), 0, bins - 1)
    error = 0.0
    for current in range(bins):
        mask = bin_id == current
        if mask.any():
            error += mask.mean() * abs(y[mask].mean() - probability[mask].mean())
    return float(error)


def maximum_calibration_error(
    y_true: Iterable[float], y_probability: Iterable[float], *, bins: int = 10
) -> float:
    y = np.asarray(list(y_true), dtype=float)
    probability = np.asarray(list(y_probability), dtype=float)
    _validate_probabilities(y, probability)
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_id = np.clip(np.digitize(probability, edges[1:-1], right=False), 0, bins - 1)
    errors = [
        abs(y[bin_id == current].mean() - probability[bin_id == current].mean())
        for current in range(bins)
        if (bin_id == current).any()
    ]
    return float(max(errors, default=0.0))


def _validate_probabilities(y: np.ndarray, probability: np.ndarray) -> None:
    if y.shape != probability.shape or y.ndim != 1 or len(y) == 0:
        raise ValueError("targets and probabilities must be non-empty one-dimensional arrays")
    if not np.isfinite(y).all() or not np.isfinite(probability).all():
        raise ValueError("targets and probabilities must be finite")
    if not np.isin(y, [0.0, 1.0]).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("targets must be binary and probabilities must be in [0, 1]")


def probability_metrics(
    y_true: Iterable[float], y_probability: Iterable[float], *, threshold: float = 0.5
) -> dict[str, float | int]:
    y = np.asarray(list(y_true), dtype=float)
    probability = np.asarray(list(y_probability), dtype=float)
    _validate_probabilities(y, probability)
    clipped = np.clip(probability, 1e-12, 1 - 1e-12)
    metrics: dict[str, float | int] = {
        "n": int(len(y)),
        "base_rate": float(y.mean()),
        "mean_probability": float(probability.mean()),
        "accuracy": float(accuracy_score(y, probability >= threshold)),
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, clipped, labels=[0, 1])),
        "ece_10": expected_calibration_error(y, probability, bins=10),
        "mce_10": maximum_calibration_error(y, probability, bins=10),
    }
    metrics["roc_auc"] = (
        float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else float("nan")
    )
    return metrics


def calibration_table(
    y_true: Iterable[float], y_probability: Iterable[float], *, bins: int = 10
) -> pd.DataFrame:
    y = np.asarray(list(y_true), dtype=float)
    probability = np.asarray(list(y_probability), dtype=float)
    _validate_probabilities(y, probability)
    bucket = pd.cut(probability, bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    frame = pd.DataFrame({"target": y, "probability": probability, "bucket": bucket})
    result = frame.groupby("bucket", observed=False).agg(
        count=("target", "size"),
        predicted=("probability", "mean"),
        observed=("target", "mean"),
    )
    result["calibration_error"] = (result["predicted"] - result["observed"]).abs()
    return result.reset_index()


def suspicious_feature_names(columns: Iterable[str]) -> list[str]:
    """Flag outcome-like names for manual point-in-time review."""

    tokens = (
        "home_score",
        "away_score",
        "actual_",
        "target_",
        "bet_return",
        "covered",
        "overhit",
        "underhit",
        "winner",
        "result",
        "total_score",
    )
    return sorted(
        column for column in columns if any(token in column.lower() for token in tokens)
    )


def assert_no_future_results(
    frame: pd.DataFrame,
    *,
    cutoff_at: object,
    time_column: str = "gameday",
    outcome_columns: Iterable[str] = ("home_score", "away_score"),
) -> None:
    require_columns(frame, (time_column, *outcome_columns), context="assert_no_future_results")
    time = pd.to_datetime(frame[time_column], utc=True, errors="raise")
    cutoff = pd.Timestamp(cutoff_at)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    future = time > cutoff
    exposed = frame.loc[future, list(outcome_columns)].notna().any(axis=1)
    if exposed.any():
        raise SchemaError(f"{int(exposed.sum())} future games expose outcome data")
