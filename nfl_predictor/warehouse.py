"""SQLite helpers for the normalized, append-oriented v2 data model."""

from __future__ import annotations

import hashlib
import re
import sqlite3
from pathlib import Path

import pandas as pd


SCHEMA_PATH = Path(__file__).with_name("schema_v2.sql")
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def connect(path: str | Path) -> sqlite3.Connection:
    if str(path) == ":memory:":
        destination: str | Path = ":memory:"
    else:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(destination)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def initialize(path: str | Path) -> None:
    with connect(path) as connection:
        connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))


def append_frame(connection: sqlite3.Connection, table: str, frame: pd.DataFrame) -> int:
    """Append rows transactionally after validating the SQL identifier."""

    if not _IDENTIFIER.fullmatch(table):
        raise ValueError("invalid SQL table name")
    if frame.empty:
        return 0
    with connection:
        frame.to_sql(table, connection, if_exists="append", index=False, method="multi")
    return int(len(frame))


def content_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def foreign_key_violations(connection: sqlite3.Connection) -> pd.DataFrame:
    rows = connection.execute("PRAGMA foreign_key_check").fetchall()
    return pd.DataFrame(rows, columns=["table", "rowid", "parent", "foreign_key_index"])
