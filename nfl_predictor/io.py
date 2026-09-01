"""Adapters for the repository's legacy tab-separated CSV artifacts."""

from __future__ import annotations

import json
import gzip
import os
from pathlib import Path
from typing import Any

import pandas as pd


def detect_delimiter(path: str | Path) -> str:
    source = Path(path)
    if source.suffix == ".gz":
        handle = gzip.open(source, "rt", encoding="utf-8", errors="replace")
    else:
        handle = source.open("rt", encoding="utf-8", errors="replace")
    with handle:
        header = handle.readline()
    return "\t" if header.count("\t") > header.count(",") else ","


def read_table(path: str | Path, *, low_memory: bool = False) -> pd.DataFrame:
    source = Path(path)
    compression = "gzip" if source.suffix == ".gz" else "infer"
    return pd.read_csv(
        source,
        sep=detect_delimiter(source),
        compression=compression,
        low_memory=low_memory,
    )


def write_table(frame: pd.DataFrame, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".tmp" + destination.suffix)
    if destination.suffix == ".parquet":
        frame.to_parquet(temporary, index=False)
    else:
        frame.to_csv(temporary, index=False)
    os.replace(temporary, destination)


def write_json(payload: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    os.replace(temporary, destination)


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"cannot serialize {type(value).__name__}")
