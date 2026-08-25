"""Atomic artifact publication and immutable prediction-journal writes."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from uuid import uuid4

import pandas as pd

from .contracts import PREDICTION_SNAPSHOT_CONTRACT
from .ingestion import utc_now


def atomic_write_json(payload: object, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(temporary, destination)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def publish_manifest(
    path: str | Path,
    *,
    artifact_type: str,
    source_run_ids: list[str],
    feature_set_version: str,
    cutoff_at: object,
    model_run_id: str | None = None,
    metrics: dict[str, object] | None = None,
) -> dict[str, object]:
    manifest = {
        "artifact_type": artifact_type,
        "created_at": utc_now(),
        "code_commit": git_commit(),
        "source_run_ids": sorted(source_run_ids),
        "feature_set_version": feature_set_version,
        "cutoff_at": str(cutoff_at),
        "model_run_id": model_run_id,
        "metrics": metrics or {},
    }
    atomic_write_json(manifest, path)
    return manifest


def create_model_run(connection, *, model_name: str, model_version: str, feature_set_version: str, target_name: str, train_start_at: object, train_end_at: object, calibration_start_at: object | None, calibration_end_at: object | None, parameters: dict[str, object], metrics: dict[str, object]) -> str:
    model_run_id = str(uuid4())
    connection.execute(
        """INSERT INTO model_run
           (model_run_id, model_name, model_version, feature_set_version, target_name,
            train_start_at, train_end_at, calibration_start_at, calibration_end_at,
            code_commit, parameters_json, metrics_json, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (model_run_id, model_name, model_version, feature_set_version, target_name,
         str(train_start_at), str(train_end_at), None if calibration_start_at is None else str(calibration_start_at),
         None if calibration_end_at is None else str(calibration_end_at), git_commit(),
         json.dumps(parameters, sort_keys=True), json.dumps(metrics, sort_keys=True), utc_now()),
    )
    return model_run_id


def publish_predictions(connection, predictions: pd.DataFrame) -> int:
    """Append validated prediction snapshots; historical keys cannot be overwritten."""

    candidate = predictions.copy()
    if "prediction_id" not in candidate:
        candidate["prediction_id"] = [str(uuid4()) for _ in range(len(candidate))]
    elif candidate["prediction_id"].isna().any():
        candidate.loc[candidate["prediction_id"].isna(), "prediction_id"] = [
            str(uuid4()) for _ in range(int(candidate["prediction_id"].isna().sum()))
        ]
    checked = PREDICTION_SNAPSHOT_CONTRACT.validate(candidate)
    with connection:
        checked.to_sql("prediction_snapshot", connection, if_exists="append", index=False, method="multi")
    return int(len(checked))


def record_shadow_decisions(connection, decisions: pd.DataFrame, *, policy_version: str) -> int:
    """Append passes and flat-stake shadow decisions tied to quoted snapshots."""

    required = {
        "prediction_id", "market_snapshot_id", "decision", "edge", "expected_profit_per_unit", "stake_fraction",
    }
    missing = sorted(required - set(decisions.columns))
    if missing:
        raise ValueError(f"shadow decisions missing columns: {missing}")
    payload = decisions[list(required)].copy()
    if not payload["decision"].isin(["pass", "shadow"]).all():
        raise ValueError("shadow journal only accepts pass or shadow decisions")
    payload["bet_decision_id"] = [str(uuid4()) for _ in range(len(payload))]
    payload["decided_at"] = utc_now()
    payload["policy_version"] = policy_version
    with connection:
        payload.to_sql("bet_decision", connection, if_exists="append", index=False, method="multi")
    return int(len(payload))
