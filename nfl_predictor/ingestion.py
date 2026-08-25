"""Point-in-time source ingestion for the V2 warehouse.

This module deliberately accepts local, versioned files.  Network acquisition is
kept in scheduled jobs so page rendering and test runs cannot silently fetch or
overwrite historical information.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4, uuid5, NAMESPACE_URL

import pandas as pd

from .contracts import GAME_FACT_CONTRACT, MARKET_SNAPSHOT_CONTRACT, SchemaError
from .warehouse import append_frame, content_sha256


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def start_source_run(
    connection,
    *,
    source_name: str,
    source_uri: str | None,
    available_at: object,
    content_hash: str | None = None,
) -> str:
    source_run_id = str(uuid4())
    connection.execute(
        """INSERT INTO source_run
           (source_run_id, source_name, source_uri, started_at, available_at, content_sha256, status)
           VALUES (?, ?, ?, ?, ?, ?, 'running')""",
        (source_run_id, source_name, source_uri, utc_now(), _timestamp(available_at), content_hash),
    )
    return source_run_id


def finish_source_run(connection, source_run_id: str, *, row_count: int, error: Exception | None = None) -> None:
    status = "failed" if error else "succeeded"
    connection.execute(
        """UPDATE source_run
           SET completed_at = ?, row_count = ?, status = ?, error_message = ?
           WHERE source_run_id = ?""",
        (utc_now(), int(row_count), status, None if error is None else str(error), source_run_id),
    )


def record_quality_event(
    connection,
    *,
    table_name: str,
    severity: str,
    rule_name: str,
    affected_rows: int,
    source_run_id: str | None = None,
    examples_json: str | None = None,
) -> None:
    connection.execute(
        """INSERT INTO data_quality_event
           (data_quality_event_id, source_run_id, table_name, severity, rule_name,
            affected_rows, examples_json, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (str(uuid4()), source_run_id, table_name, severity, rule_name, int(affected_rows), examples_json, utc_now()),
    )


def ingest_games(connection, frame: pd.DataFrame, *, source_run_id: str) -> int:
    games = GAME_FACT_CONTRACT.validate(frame)
    result = pd.DataFrame(
        {
            "game_id": games["game_id"].astype(str),
            "season": games["season"].astype(int),
            "week": games["week"].astype(int),
            "game_type": games.get("game_type"),
            "kickoff_at": _kickoff_at(games),
            "home_team": games["home_team"].astype(str),
            "away_team": games["away_team"].astype(str),
            "neutral_site": _optional_numeric(games, "neutral_site", default=0).fillna(0).astype(int),
            "stadium_id": games.get("stadium_id", games.get("stadium")),
            "roof": games.get("roof"),
            "surface": games.get("surface"),
            "home_score": _optional_numeric(games, "home_score"),
            "away_score": _optional_numeric(games, "away_score"),
            "result_available_at": _optional_timestamp(games.get("result_available_at")),
            "source_run_id": source_run_id,
        }
    )
    statement = """INSERT INTO game
        (game_id, season, week, game_type, kickoff_at, home_team, away_team, neutral_site,
         stadium_id, roof, surface, home_score, away_score, result_available_at, source_run_id)
        VALUES (:game_id, :season, :week, :game_type, :kickoff_at, :home_team, :away_team, :neutral_site,
         :stadium_id, :roof, :surface, :home_score, :away_score, :result_available_at, :source_run_id)
        ON CONFLICT(game_id) DO NOTHING"""
    before = connection.total_changes
    with connection:
        connection.executemany(statement, result.where(pd.notna(result), None).to_dict("records"))
    inserted = connection.total_changes - before
    duplicates = len(result) - int(inserted)
    if duplicates:
        record_quality_event(connection, table_name="game", severity="warning", rule_name="existing_game_ignored", affected_rows=duplicates, source_run_id=source_run_id)
    return int(inserted)


def ingest_market_snapshots(connection, frame: pd.DataFrame, *, source_run_id: str) -> int:
    markets = MARKET_SNAPSHOT_CONTRACT.validate(frame)
    result = markets.copy()
    result["market_snapshot_id"] = [
        str(uuid5(NAMESPACE_URL, "|".join(map(str, row))))
        for row in result[["game_id", "book", "market", "participant_id", "side", "line", "observed_at"]].itertuples(index=False, name=None)
    ]
    result["source_run_id"] = source_run_id
    for column in ("observed_at", "available_at"):
        result[column] = result[column].map(_timestamp)
    append_frame(connection, "market_snapshot", result[[
        "market_snapshot_id", "game_id", "book", "market", "participant_id", "side", "line",
        "price_american", "observed_at", "available_at", "source_run_id",
    ]])
    return int(len(result))


def ingest_file(
    connection,
    *,
    path: str | Path,
    source_name: str,
    source_uri: str | None,
    available_at: object,
    kind: str,
) -> tuple[str, int]:
    """Register one immutable source file and load its normalized facts."""

    from .io import read_table

    source = Path(path)
    run_id = start_source_run(connection, source_name=source_name, source_uri=source_uri, available_at=available_at, content_hash=content_sha256(source))
    try:
        frame = read_table(source)
        loader = {"games": ingest_games, "markets": ingest_market_snapshots}.get(kind)
        if loader is None:
            raise SchemaError("kind must be 'games' or 'markets'")
        rows = loader(connection, frame, source_run_id=run_id)
    except Exception as exc:
        finish_source_run(connection, run_id, row_count=0, error=exc)
        record_quality_event(connection, table_name=kind, severity="error", rule_name="ingestion_failed", affected_rows=0, source_run_id=run_id, examples_json=repr(str(exc)))
        raise
    finish_source_run(connection, run_id, row_count=rows)
    return run_id, rows


def _timestamp(value: object) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat().replace("+00:00", "Z")


def _optional_timestamp(values: object) -> pd.Series | None:
    if values is None:
        return None
    converted = pd.to_datetime(values, utc=True, errors="coerce")
    return converted.map(lambda value: None if pd.isna(value) else value.isoformat().replace("+00:00", "Z"))


def _optional_numeric(frame: pd.DataFrame, column: str, *, default: float | None = None) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _kickoff_at(games: pd.DataFrame) -> pd.Series:
    if "kickoff_at" in games:
        return pd.to_datetime(games["kickoff_at"], utc=True, errors="raise").map(_timestamp)
    # NFLverse gametime is Eastern local time; UTC conversion makes comparisons unambiguous.
    date = games["gameday"].dt.strftime("%Y-%m-%d")
    time = games.get("gametime", pd.Series("00:00", index=games.index)).fillna("00:00")
    return pd.to_datetime(date + " " + time, errors="raise").dt.tz_localize("America/New_York").dt.tz_convert("UTC").map(_timestamp)
