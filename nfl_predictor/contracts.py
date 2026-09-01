"""Data contracts and point-in-time validation.

The legacy project stores many concepts in one wide CSV.  These contracts make
availability time, uniqueness, and nullable outcomes explicit before a frame is
allowed into feature engineering or evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import pandas as pd


class SchemaError(ValueError):
    """Raised when a dataframe violates a declared data contract."""


@dataclass(frozen=True)
class TableContract:
    """Lightweight dataframe contract with no third-party validation dependency."""

    name: str
    required: tuple[str, ...]
    key: tuple[str, ...] = ()
    datetime_columns: tuple[str, ...] = ()
    numeric_columns: tuple[str, ...] = ()
    allowed_values: Mapping[str, frozenset[object]] = field(default_factory=dict)

    def validate(self, frame: pd.DataFrame, *, allow_empty: bool = False) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame):
            raise SchemaError(f"{self.name}: expected pandas.DataFrame")
        if frame.empty and not allow_empty:
            raise SchemaError(f"{self.name}: dataframe is empty")

        missing = sorted(set(self.required) - set(frame.columns))
        if missing:
            raise SchemaError(f"{self.name}: missing required columns: {missing}")

        result = frame.copy()
        for column in self.datetime_columns:
            converted = pd.to_datetime(result[column], utc=True, errors="coerce")
            bad = result[column].notna() & converted.isna()
            if bad.any():
                raise SchemaError(
                    f"{self.name}.{column}: {int(bad.sum())} values are not timestamps"
                )
            result[column] = converted

        for column in self.numeric_columns:
            converted = pd.to_numeric(result[column], errors="coerce")
            bad = result[column].notna() & converted.isna()
            if bad.any():
                raise SchemaError(
                    f"{self.name}.{column}: {int(bad.sum())} values are not numeric"
                )
            result[column] = converted

        if self.key:
            null_key = result[list(self.key)].isna().any(axis=1)
            if null_key.any():
                raise SchemaError(f"{self.name}: {int(null_key.sum())} rows have a null key")
            duplicate = result.duplicated(list(self.key), keep=False)
            if duplicate.any():
                examples = result.loc[duplicate, list(self.key)].head(3).to_dict("records")
                raise SchemaError(f"{self.name}: duplicate key values, examples={examples}")

        for column, allowed in self.allowed_values.items():
            invalid = result[column].notna() & ~result[column].isin(allowed)
            if invalid.any():
                examples = result.loc[invalid, column].drop_duplicates().head(5).tolist()
                raise SchemaError(
                    f"{self.name}.{column}: values outside contract, examples={examples}"
                )
        return result


GAME_FACT_CONTRACT = TableContract(
    name="game_fact",
    required=(
        "game_id",
        "season",
        "week",
        "gameday",
        "home_team",
        "away_team",
    ),
    key=("game_id",),
    datetime_columns=("gameday",),
    numeric_columns=("season", "week"),
)

MARKET_SNAPSHOT_CONTRACT = TableContract(
    name="market_snapshot",
    required=(
        "game_id",
        "book",
        "market",
        "participant_id",
        "side",
        "line",
        "price_american",
        "observed_at",
        "available_at",
    ),
    key=("game_id", "book", "market", "participant_id", "side", "line", "observed_at"),
    datetime_columns=("observed_at", "available_at"),
    numeric_columns=("line", "price_american"),
    allowed_values={
        "market": frozenset({"moneyline", "spread", "total", "player_prop"}),
    },
)

PREDICTION_SNAPSHOT_CONTRACT = TableContract(
    name="prediction_snapshot",
    required=(
        "prediction_id",
        "model_run_id",
        "game_id",
        "market",
        "participant_id",
        "side",
        "line",
        "model_probability",
        "cutoff_at",
        "created_at",
    ),
    key=("prediction_id",),
    datetime_columns=("cutoff_at", "created_at"),
    numeric_columns=("line", "model_probability"),
)


def assert_point_in_time(
    frame: pd.DataFrame,
    *,
    cutoff_column: str = "cutoff_at",
    available_column: str = "available_at",
) -> None:
    """Reject observations that became available after the prediction cutoff."""

    missing = {cutoff_column, available_column} - set(frame.columns)
    if missing:
        raise SchemaError(f"point-in-time check missing columns: {sorted(missing)}")
    cutoff = pd.to_datetime(frame[cutoff_column], utc=True, errors="coerce")
    available = pd.to_datetime(frame[available_column], utc=True, errors="coerce")
    invalid_timestamp = cutoff.isna() | available.isna()
    if invalid_timestamp.any():
        raise SchemaError(
            f"point-in-time check found {int(invalid_timestamp.sum())} invalid timestamps"
        )
    leaked = available > cutoff
    if leaked.any():
        columns = [column for column in ("game_id", available_column, cutoff_column) if column in frame]
        examples = frame.loc[leaked, columns].head(5).to_dict("records")
        raise SchemaError(
            f"{int(leaked.sum())} observations were unavailable at prediction time; "
            f"examples={examples}"
        )


def require_columns(frame: pd.DataFrame, columns: Iterable[str], *, context: str) -> None:
    """Raise a useful error when a transformation's inputs are incomplete."""

    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise SchemaError(f"{context}: missing required columns: {missing}")
