"""Command-line entry points for v2 feature builds and walk-forward audits."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .features import build_pregame_features, feature_columns
from .io import read_table, write_json, write_table
from .ingestion import ingest_file
from .publication import publish_manifest
from .research import benchmark_by_season, benchmark_report
from .modeling import IsotonicProbabilityCalibrator, ScoreDistributionForecaster
from .validation import probability_metrics, season_walk_forward_splits
from .warehouse import initialize
from .warehouse import connect, foreign_key_violations


DEFAULT_GAMES = Path("data_files/nfl_games_historical.csv")


def build_features(args: argparse.Namespace) -> int:
    games = read_table(args.games)
    pbp = read_table(args.pbp) if args.pbp else None
    features = build_pregame_features(games, pbp=pbp)
    write_table(features, args.output)
    print(
        f"wrote {len(features):,} rows and {len(feature_columns(features)):,} eligible features "
        f"to {args.output}"
    )
    return 0


def init_database(args: argparse.Namespace) -> int:
    initialize(args.database)
    print(f"initialized v2 warehouse at {args.database}")
    return 0


def ingest(args: argparse.Namespace) -> int:
    initialize(args.database)
    with connect(args.database) as connection:
        run_id, rows = ingest_file(
            connection,
            path=args.input,
            source_name=args.source_name,
            source_uri=args.source_uri,
            available_at=args.available_at,
            kind=args.kind,
        )
        violations = foreign_key_violations(connection)
        if not violations.empty:
            raise RuntimeError(f"foreign-key violations after ingestion: {violations.to_dict('records')}")
    print(f"ingested {rows:,} {args.kind} rows in source run {run_id}")
    return 0


def walk_forward(args: argparse.Namespace) -> int:
    games = read_table(args.games)
    frame = build_pregame_features(games)
    completed = frame[
        frame["is_completed"]
        & frame["home_margin"].notna()
        & frame["actual_total"].notna()
        & frame["spread_line"].notna()
        & frame["total_line"].notna()
    ].copy()
    eligible = feature_columns(completed)
    folds = season_walk_forward_splits(
        completed,
        first_test_season=args.first_test_season,
        calibration_seasons=1,
        minimum_train_seasons=2,
    )
    prediction_frames: list[pd.DataFrame] = []
    reports: list[dict[str, object]] = []
    for fold in folds:
        train = completed.loc[fold.train_index]
        calibration = completed.loc[fold.calibration_index]
        test = completed.loc[fold.test_index]
        model = ScoreDistributionForecaster(random_state=args.random_state)
        model.fit(train[eligible], train["home_margin"], train["actual_total"])

        calibration_sim = model.predict_distribution(
            calibration[eligible],
            spread_line=calibration["spread_line"],
            total_line=calibration["total_line"],
            simulations=args.simulations,
            random_state=args.random_state + fold.test_season,
        )
        test_sim = model.predict_distribution(
            test[eligible],
            spread_line=test["spread_line"],
            total_line=test["total_line"],
            simulations=args.simulations,
            random_state=args.random_state + fold.test_season + 1,
        )
        targets = {
            "home_win": ("target_home_win", calibration_sim.home_win_probability, test_sim.home_win_probability),
            "home_cover": ("target_home_cover", calibration_sim.home_cover_probability, test_sim.home_cover_probability),
            "over": ("target_over", calibration_sim.over_probability, test_sim.over_probability),
        }
        fold_predictions = test[["game_id", "season", "week", "gameday"]].copy()
        fold_report: dict[str, object] = {
            "test_season": fold.test_season,
            "train_rows": len(train),
            "calibration_rows": len(calibration),
            "test_rows": len(test),
        }
        for name, (target_column, calibration_probability, test_probability) in targets.items():
            calibration_mask = calibration[target_column].notna().to_numpy()
            test_mask = test[target_column].notna().to_numpy()
            calibrator = IsotonicProbabilityCalibrator().fit(
                np.asarray(calibration_probability)[calibration_mask],
                calibration.loc[calibration_mask, target_column],
            )
            calibrated = calibrator.predict(test_probability)
            fold_predictions[f"prob_{name}"] = calibrated
            fold_predictions[f"target_{name}"] = test[target_column].to_numpy()
            fold_report[name] = probability_metrics(
                test.loc[test_mask, target_column], calibrated[test_mask]
            )
        prediction_frames.append(fold_predictions)
        reports.append(fold_report)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    write_table(predictions, args.predictions_output)
    write_json(
        {
            "method": "expanding-season walk-forward with held-out isotonic calibration",
            "eligible_feature_count": len(eligible),
            "eligible_features": eligible,
            "folds": reports,
        },
        args.metrics_output,
    )
    if args.manifest_output:
        publish_manifest(
            args.manifest_output,
            artifact_type="walk_forward_predictions",
            source_run_ids=args.source_run_id,
            feature_set_version=args.feature_set_version,
            cutoff_at="close_benchmark",
            metrics={"prediction_rows": len(predictions), "folds": len(reports)},
        )
    print(f"wrote {len(predictions):,} out-of-sample predictions")
    return 0


def benchmark(args: argparse.Namespace) -> int:
    predictions = read_table(args.predictions)
    report = {"overall": benchmark_report(predictions), "by_season": benchmark_by_season(predictions)}
    write_json(report, args.output)
    print(f"wrote predeclared benchmark report to {args.output}")
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="NFL predictor v2 tools")
    commands = root.add_subparsers(dest="command", required=True)

    features = commands.add_parser("build-features", help="build point-in-time game features")
    features.add_argument("--games", type=Path, default=DEFAULT_GAMES)
    features.add_argument("--pbp", type=Path)
    features.add_argument("--output", type=Path, default=Path("data_files/v2/game_features.csv"))
    features.set_defaults(handler=build_features)

    database = commands.add_parser("init-db", help="initialize the normalized SQLite warehouse")
    database.add_argument("--database", type=Path, default=Path("data_files/v2/nfl_v2.sqlite3"))
    database.set_defaults(handler=init_database)

    loader = commands.add_parser("ingest", help="register a local source file and load normalized facts")
    loader.add_argument("--database", type=Path, default=Path("data_files/v2/nfl_v2.sqlite3"))
    loader.add_argument("--input", type=Path, required=True)
    loader.add_argument("--kind", choices=("games", "markets"), required=True)
    loader.add_argument("--source-name", required=True)
    loader.add_argument("--source-uri")
    loader.add_argument("--available-at", required=True, help="UTC timestamp when this file became usable")
    loader.set_defaults(handler=ingest)

    backtest = commands.add_parser("walk-forward", help="run expanding-season score backtest")
    backtest.add_argument("--games", type=Path, default=DEFAULT_GAMES)
    backtest.add_argument("--first-test-season", type=int)
    backtest.add_argument("--simulations", type=int, default=5_000)
    backtest.add_argument("--random-state", type=int, default=42)
    backtest.add_argument(
        "--predictions-output",
        type=Path,
        default=Path("data_files/v2/walk_forward_predictions.csv"),
    )
    backtest.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("data_files/v2/walk_forward_metrics.json"),
    )
    backtest.add_argument("--source-run-id", action="append", default=[])
    backtest.add_argument("--feature-set-version", default="v2.game_features")
    backtest.add_argument("--manifest-output", type=Path)
    backtest.set_defaults(handler=walk_forward)

    benchmark_command = commands.add_parser("benchmark", help="report model versus declared market baselines")
    benchmark_command.add_argument("--predictions", type=Path, required=True)
    benchmark_command.add_argument("--output", type=Path, default=Path("data_files/v2/benchmark_report.json"))
    benchmark_command.set_defaults(handler=benchmark)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
