# V2 implementation and migration guide

> **Implementation review — 2026-08-24.** The files listed below, the SQLite schema, CLI, tests, and `v2-quality.yml` workflow are present in the working tree. Migration steps 1–8 are **not complete**: the implementation is parallel to the legacy app, with no observed production source manifests/loaders, published prediction snapshots, decision journal, or Streamlit integration. The test command requires the declared runtime dependencies; in the bundled review interpreter `scikit-learn` was absent, so four V2 test modules could not import. This is an environment verification gap, not evidence that those tests pass locally.

The v2 package is a parallel research path. It does not replace `predictions.py` or write over the legacy model artifacts unless an operator explicitly chooses those output paths.

## Files added

```text
nfl_predictor/
    __init__.py          public package surface
    contracts.py         dataframe contracts and availability checks
    features.py          prior-only game and PBP feature engineering
    markets.py           odds, de-vig, EV, Kelly, CLV, settlement
    validation.py        walk-forward folds and probability metrics
    modeling.py          Elo, held-out calibration, score simulation
    backtest.py          bet policy, grading, bankroll, uncertainty, gates
    io.py                legacy CSV adapters
    warehouse.py         SQLite initialization and append helpers
    schema_v2.sql        normalized warehouse DDL
    cli.py               executable workflows
scripts/
    nfl_v2.py            thin CLI wrapper
tests/
    test_v2_*.py         deterministic core tests
```

## Run the tests

The v2 tests use the Python standard library's `unittest`; pytest is not required.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:LOKY_MAX_CPU_COUNT='4'
venv\Scripts\python.exe -m unittest discover -s tests -p 'test_v2_*.py' -v
```

The suite covers contracts, point-in-time enforcement, target and spread semantics, future-game handling, 30+ feature admission, weather correctness, odds/de-vig/EV/Kelly, pushes, settlement, walk-forward ordering, calibration metrics, Elo, score simulation, bankroll reporting, and promotion gates.

## Build point-in-time features

Game-level only:

```powershell
venv\Scripts\python.exe scripts\nfl_v2.py build-features `
  --games data_files\nfl_games_historical.csv `
  --output data_files\v2\game_features.csv
```

Game plus play-by-play:

```powershell
venv\Scripts\python.exe scripts\nfl_v2.py build-features `
  --games data_files\nfl_games_historical.csv `
  --pbp data_files\nfl_play_by_play_historical.csv.gz `
  --output data_files\v2\game_features_with_pbp.csv
```

CSV is the default because the current requirements do not declare a Parquet engine. Install and pin `pyarrow` before choosing a `.parquet` output.

## Initialize the warehouse

```powershell
venv\Scripts\python.exe scripts\nfl_v2.py init-db `
  --database data_files\v2\nfl_v2.sqlite3
```

The database is ignored by the existing `*.sqlite3` rule. Production should promote immutable warehouse snapshots or source-run manifests as workflow artifacts rather than commit a mutable database to Git.

## Run the season-forward replay

```powershell
venv\Scripts\python.exe scripts\nfl_v2.py walk-forward `
  --games data_files\nfl_games_historical.csv `
  --first-test-season 2023 `
  --simulations 5000 `
  --predictions-output data_files\v2\walk_forward_predictions.csv `
  --metrics-output data_files\v2\walk_forward_metrics.json
```

For each test season the workflow:

1. trains only on seasons before the calibration season;
2. fits the joint margin/total model;
3. simulates probabilities on the calibration season;
4. fits isotonic calibration on that held-out season;
5. makes one untouched set of predictions for the test season;
6. writes Brier, log loss, AUC, accuracy, ECE, and MCE;
7. preserves per-game out-of-sample probabilities.

This command is an honest starting point, not automatic proof of an edge. Add market-only, Elo-only, and ablation comparisons before promotion.

## Migration sequence

### 1. Freeze the legacy baseline

- Keep `data_files/model_metrics.json` and the current model CSV as historical artifacts.
- Record that the saved spread policy is 28-26 with -1.01% theoretical ROI.
- Do not rebrand legacy in-sample probabilities as out-of-sample.

### 2. Create source manifests

For every schedule, play-by-play, roster, injury, snap, and odds fetch:

- create a `source_run_id`;
- record source URL/version and license;
- record start, completion, observed, and availability timestamps;
- hash the downloaded bytes;
- record row count and status;
- persist failures as failures, not empty successful datasets.

### 3. Populate normalized facts

- Load games first.
- Expand games into two `team_game` rows after completion.
- Load player opportunity facts using GSIS IDs.
- Append market and availability snapshots; never update historical quotes in place.
- run `PRAGMA foreign_key_check` after every load batch.

### 4. Materialize features at declared horizons

Recommended cutoffs:

- **early:** Tuesday after the prior week is final;
- **midweek:** after Wednesday practice reports;
- **late:** 90 minutes before kickoff/inactives;
- **close benchmark:** last broadly available quote before kickoff, never an early-pick input.

Every horizon receives a distinct `feature_set_version` or cutoff field.

### 5. Add declared baselines

- naive home base rate;
- Elo;
- no-vig market moneyline;
- spread as predicted margin;
- total line as predicted total;
- simple rolling EPA model;
- current gradient-boosting model.

A complex model should not ship when it fails to improve probability score, point error, or economic value over these baselines.

### 6. Add a prediction publisher

The training job should publish:

- model-run record;
- feature list and schema hash;
- source-run IDs and hashes;
- code commit;
- training/calibration cutoffs;
- raw, calibrated, and market probabilities;
- prediction intervals;
- model card and known limitations.

Streamlit should read that immutable artifact. It should not train or settle while rendering a page.

### 7. Add paper decisions and settlements

- Log every bet/pass/shadow decision with the exact quote.
- Grade after a final result is available.
- Preserve push and void states.
- capture closing quote and CLV.
- use flat paper stakes until gates pass.
- calculate season-clustered uncertainty and portfolio drawdown.

### 8. Refactor the dashboard incrementally

Suggested modules:

```text
app/
    data_service.py
    prediction_service.py
    market_service.py
    settlement_service.py
    export_service.py
    pages/
```

Start by extracting read-only loaders and pure formatting. Then move networking and writes into scheduled jobs. Finally, replace legacy model fields with published v2 prediction snapshots.

## CI changes

Add one non-network quality workflow on pushes and pull requests:

```yaml
- run: python -m compileall nfl_predictor tests
- run: python -m unittest discover -s tests -p "test_v2_*.py" -v
- run: python scripts/nfl_v2.py build-features --output build/game_features.csv
```

Then add separately pinned lint, dependency audit, secret scanning, and a small fixture-based walk-forward smoke test. Network ingestion belongs in scheduled workflows, not the unit-test job.

For nightly jobs:

- fail when a required source is stale or empty;
- publish one artifact bundle and manifest;
- use concurrency controls;
- open an automated data-update PR or promote through a single writer;
- do not allow several workflows to push generated files to `main` independently.

## Verification completed on 2026-08-11

The checked-in code was exercised against `data_files/nfl_games_historical.csv`, not only test fixtures:

- 1,693 game rows across six seasons built successfully;
- 194 eligible numeric pregame features remained after outcomes and provider identifiers were denied;
- the compressed historical PBP delimiter was detected correctly;
- all 12 normalized SQLite tables initialized in memory with zero foreign-key violations;
- 31 deterministic V2 tests pass in the pinned V2 environment (`requirements-v2.txt`);
- the 2023-2025 expanding-season replay produced 855 strictly out-of-sample predictions with a separate prior-season calibration fold.

Smoke-test Brier scores (1,000 simulation draws per game) were:

| Test season | Home win | Home cover | Over |
|---:|---:|---:|---:|
| 2023 | 0.2343 | 0.2476 | 0.2489 |
| 2024 | 0.2378 | 0.2549 | 0.2564 |
| 2025 | 0.2165 | 0.2455 | 0.2480 |

These numbers establish that the replay executes and probabilities are auditable. They do not establish a profitable betting policy; price-aware decision and CLV gates still apply.

## Production gates

Minimum game-market gate:

- two forward seasons untouched by model/policy design;
- 500+ settled decisions for the promoted policy;
- positive mean CLV;
- positive ROI after actual vig and best-price assumptions that can be reproduced;
- calibration slope/intercept and reliability table within declared tolerance;
- no single season responsible for the entire result;
- acceptable maximum drawdown under flat stakes;
- no critical data-quality event in the promoted period.

Minimum player-prop gate:

- player-forward and season-forward splits;
- separate opportunity and efficiency models;
- book-specific lines, prices, and void rules;
- active/snap uncertainty captured before decision;
- calibration by prop, line band, position, and side;
- correlated exposure limits.

## Operational checks before each slate

- Is every game keyed uniquely?
- Is the schedule current and kickoff timezone correct?
- Are source hashes and row counts plausible?
- Are odds recent enough and present on both sides?
- Was overround removed from a complete market?
- Are quarterback/inactive scenarios unresolved?
- Is weather a forecast with issue time, or a stale cached value?
- Does every feature satisfy `available_at <= cutoff_at`?
- Does the prediction reference a valid model run?
- Is the decision policy version recorded?

If any answer is no, the correct output is `NO BET / DATA UNRESOLVED`, not a fallback confidence label.
