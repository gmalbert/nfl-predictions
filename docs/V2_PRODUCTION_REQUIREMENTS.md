# V2 Production Requirements and Implementation Guide

> **Prepared:** 2026-08-24  
> **Purpose:** Define what remains before V2 forecasts can move from research/shadow mode to production-facing recommendations.

## Current capability

The repository now has a committed V2 research foundation:

- point-in-time contracts and prior-only features;
- normalized SQLite schema;
- season-forward replay, calibration, market math, settlement, and promotion-gate primitives;
- local source-file ingestion with hashes, timestamps, and quality events;
- atomic artifact manifests and database-backed shadow/pass decisions;
- pinned V2 dependencies and deterministic CI tests.

The legacy Streamlit application remains research-only. It must not treat confidence tiers, ROI, or bankroll recommendations as validated betting advice until the release gate below passes.

## Required external decisions

These choices require an operator decision because they determine licensing, availability, and reproducibility.

| Requirement | Decision needed | Minimum standard |
|---|---|---|
| Schedule/results source | Confirm canonical provider | NFLverse schedule/game data is acceptable if raw releases and fetch time are recorded. |
| Odds provider | Choose licensed or permitted multi-book source | Must preserve book, side, line, American price, provider-observed time, and system-available time. |
| Injury/availability source | Choose replayable injury, practice, inactive, and depth-chart source | Must preserve stable player ID, team, status, source URL, observed time, and available time. |
| Player opportunity source | Choose snap, route, target, carry, and dropback source | Must have stable player IDs and a historical source/release policy. |
| Artifact storage | Choose durable non-Git storage | Raw sources, warehouse snapshots, manifests, model artifacts, and published predictions must be retained and addressable. |
| Deployment writer | Choose one scheduled job/service | Only one writer may promote a production artifact set at a time. |

Do not substitute scraped, untimestamped, or license-restricted data without documenting permission and replay limitations.

## Required data contract

For every source acquisition, record one `source_run` row:

```text
source_run_id
source_name
source_uri
started_at
completed_at
available_at
content_sha256
row_count
status
error_message
```

Every source record used by a prediction must satisfy:

```text
available_at <= cutoff_at
```

Event date alone is insufficient. A Wednesday practice event published on Thursday cannot be used by a Wednesday prediction.

### Required warehouse tables

| Table | State | Remaining work |
|---|---|---|
| `source_run` | Implemented | Use for every scheduled source load. |
| `game` | Implemented, local CSV loader | Load canonical schedule/results and preserve kickoff time. |
| `team_game` | Schema only | Add postgame team/PBP fact loader. |
| `market_snapshot` | Implemented, local CSV loader | Add scheduled multi-book capture and complete two-sided market checks. |
| `availability_snapshot` | Schema only | Add timestamped injury/practice/inactive loader. |
| `player_game` | Schema only | Add player opportunity loader keyed by stable player ID. |
| `feature_snapshot` | Schema only | Materialize features for early, midweek, late, and close horizons. |
| `model_run` / `prediction_snapshot` | Implemented write primitives | Add scheduled model publisher. |
| `bet_decision` / `bet_settlement` | Shadow/pass writer implemented | Add final-result grading, void/push handling, close capture, and CLV. |
| `data_quality_event` | Implemented | Require monitoring and fail/hold behavior for error-severity events. |

## Required scheduled workflows

### 1. Source acquisition

Run on a schedule and at every declared decision horizon.

```powershell
python scripts/nfl_v2.py ingest `
  --database data_files/v2/nfl_v2.sqlite3 `
  --input data_files/raw/schedule_2026-09-01.csv `
  --kind games `
  --source-name nflverse_schedule `
  --source-uri https://example-provider/release `
  --available-at 2026-09-01T12:00:00Z
```

For market files, use `--kind markets`. The input must include:

```text
game_id, book, market, participant_id, side, line, price_american, observed_at, available_at
```

The job must:

1. save the raw response before parsing;
2. register the source run and SHA-256 hash;
3. validate keys, timestamps, domain values, and foreign keys;
4. create `data_quality_event` records for failures/staleness;
5. stop promotion when a required source is missing or stale.

### 2. Feature materialization

Build each horizon separately. Do not use a close quote in an early/midweek feature frame.

```powershell
python scripts/nfl_v2.py build-features `
  --games data_files/nfl_games_historical.csv `
  --pbp data_files/nfl_play_by_play_historical.csv.gz `
  --output data_files/v2/game_features.csv
```

Required extensions:

- add a feature-snapshot writer for each cutoff;
- retain the source-run IDs and feature-set version;
- store missing as missing, never as an implicit zero;
- reject rows that violate the point-in-time check.

### 3. Training and publication

The training job must write one `model_run`, then append immutable `prediction_snapshot` records. Streamlit must only read those published artifacts.

Minimum model-run metadata:

```text
model_run_id
model_name / version
feature_set_version
target
training and calibration windows
code commit
parameters
metrics
source-run IDs and hashes
```

Create an atomic manifest for every published artifact:

```python
from nfl_predictor.publication import publish_manifest

publish_manifest(
    "data_files/v2/published/run_manifest.json",
    artifact_type="prediction_snapshot",
    source_run_ids=["..."],
    feature_set_version="v2.game_features",
    cutoff_at="2026-09-10T17:30:00Z",
    model_run_id="...",
    metrics={"brier": 0.22},
)
```

### 4. Shadow decisions and settlement

Until promotion, decisions must be `pass` or `shadow` with flat paper stakes. No Kelly or bankroll percentages.

```python
from nfl_predictor.research import flat_shadow_decisions
from nfl_predictor.publication import record_shadow_decisions

decisions = flat_shadow_decisions(candidates, minimum_edge=0.025, paper_stake=1.0)
record_shadow_decisions(connection, decisions, policy_version="shadow.v1")
```

The scheduled settlement job must:

1. wait for final results;
2. settle `win`, `loss`, `push`, and `void` correctly;
3. attach the last valid pre-kickoff closing quote;
4. calculate CLV from no-vig bet-time and close probabilities;
5. never rewrite a historical prediction or decision.

## Required evaluation

Run the existing close-benchmark replay before every model/policy change:

```powershell
python scripts/nfl_v2.py walk-forward `
  --games data_files/nfl_games_historical.csv `
  --first-test-season 2023 `
  --simulations 1000 `
  --predictions-output data_files/v2/walk_forward_predictions.csv `
  --metrics-output data_files/v2/walk_forward_metrics.json `
  --manifest-output data_files/v2/walk_forward_manifest.json

python scripts/nfl_v2.py benchmark `
  --predictions data_files/v2/walk_forward_predictions.csv `
  --output data_files/v2/benchmark_report.json
```

The currently completed historical run produces 855 out-of-sample, close-benchmark predictions for 2023–2025. It is a reproducibility baseline only; it does not establish betting profitability because timestamped multi-book price snapshots are not yet available.

### Required comparisons

For every released target/horizon, report overall and by season:

- home-rate baseline;
- Elo baseline;
- de-vigged market baseline where prices exist;
- rolling-form-only model;
- PBP-only model;
- combined model;
- Brier score, log loss, AUC, ECE/MCE, and reliability table;
- margin/total error for distribution models;
- decision count, post-vig ROI, CLV, drawdown, and clustered uncertainty.

Do not select probability thresholds, feature groups, price bands, or betting slices on the final reported season.

## Release gate

A market can move from shadow to recommendation only when all conditions are met:

1. Two forward seasons have completed without changing the promoted model/policy from later results.
2. At least 500 settled decisions exist for that market, unless a documented smaller-market exception is approved.
3. Mean CLV is positive.
4. ROI is positive after actual vig and reproducible best-price assumptions.
5. A season-clustered confidence interval does not support materially negative ROI.
6. Calibration, reliability, and drawdown stay within predeclared limits.
7. No single season, book, or probability slice explains the entire result.
8. No critical data-quality event occurred in the promoted period.

The fail-closed code path is available now:

```python
from nfl_predictor.research import require_promotable_shadow_period

require_promotable_shadow_period(
    {
        "bets": 500,
        "roi": 0.02,
        "mean_clv": 0.01,
        "roi_ci_low": 0.001,
    }
)
```

## Relevant code map

| Need | Code |
|---|---|
| Point-in-time validation | `nfl_predictor/contracts.py` |
| Prior-only game/PBP features | `nfl_predictor/features.py` |
| Odds, no-vig, EV, CLV, pushes | `nfl_predictor/markets.py` |
| Model/calibration/score distribution/Elo | `nfl_predictor/modeling.py` |
| Season-forward splits and metrics | `nfl_predictor/validation.py` |
| Decision grading, drawdown, promotion utility | `nfl_predictor/backtest.py` |
| SQLite schema and helpers | `nfl_predictor/schema_v2.sql`, `nfl_predictor/warehouse.py` |
| Source ingestion / quality events | `nfl_predictor/ingestion.py` |
| Atomic artifact and shadow-journal writes | `nfl_predictor/publication.py` |
| Benchmarks and flat shadow-decision policy | `nfl_predictor/research.py` |
| CLI entry points | `scripts/nfl_v2.py`, `nfl_predictor/cli.py` |
| Pinned test environment | `requirements-v2.txt` |
| Deterministic V2 tests | `tests/test_v2_*.py` |

## Verification command

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:LOKY_MAX_CPU_COUNT='4'
python -m pip install -r requirements-v2.txt
python -m unittest discover -s tests -p 'test_v2_*.py' -v
```

The committed V2 suite currently contains 31 deterministic tests.
