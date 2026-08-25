# NFL Predictions — Architecture

> **Implementation review — 2026-08-24.** This describes the legacy architecture and is incomplete as a current-state document. A separate V2 package now provides contracts, feature engineering, markets, validation, modeling, backtesting, SQLite storage, and a CLI, but has not replaced this two-step CSV pipeline or the monolithic `predictions.py` UI. The stated “all features are pre-game only” claim is not supported for the legacy path because its evaluation remains randomly split and lacks availability timestamps. This document should be superseded by the migration priorities after V2 is integrated.

## Overview
Multi-page Streamlit app for NFL betting predictions using XGBoost models. Predicts outcomes for spread, moneyline, over/under markets, and player props. All data and models are pre-computed locally.

## Two-Step Pipeline
```
Step 1 (~5 min, run first):
    nfl_data_py library
        ↓
    create-nfl-historical.py → data_files/nfl_games_historical.csv
        ↓
    nfl-gather-data.py → Feature Engineering + XGBoost Training
        ↓
    data_files/nfl_games_historical_with_predictions.csv
    data_files/model_feature_importances.csv
    data_files/model_metrics.json

    (Run both via: python build_and_train_pipeline.py)

Step 2 — UI:
    predictions.py (main dashboard)
    pages/1_📊_Historical_Data.py
    pages/2_🎯_Player_Props.py  [includes DK Pick 6 Calculator]
    pages/3_🎲_Parlay_Builder.py
    pages/4_📈_Model_Performance.py
```

## ML Models
Three XGBoost classifiers (binary):
| Model | Target | Threshold | Notes |
|-------|--------|-----------|-------|
| Spread | `underdogCovered` | 50% | Was inverted — fixed Dec 2025 (apply `1 - prob`) |
| Moneyline | Winner | 28% | F1-optimised for underdogs |
| Totals | Over/Under | 50% | F1-optimised |

Confidence tiers: Elite ≥65%, Strong 60–65%, Good 55–60%, Standard 50–55%

### Critical: Spread Model Inversion Fix
After training spread model: `prob_underdogCovered = 1 - prob_underdogCovered`
Impact: ROI improved from -90% to +60%.

### Player Props (`player_props/`)
XGBoost + LightGBM soft-voting ensembles per stat category. Models in `player_props/models/*.json`. DK Pick 6 Calculator in `pages/2_🎯_Player_Props.py`.

## Feature Engineering
All features are pre-game only (zero data leakage):
- **Momentum** (8): Last 3 games win%, scoring, point differential
- **Rest** (5): Rest day differences, well-rested ≥10d / short-rest ≤6d flags
- **Weather** (3): Cold ≤32°F, windy ≥15mph, extreme conditions
- Rolling stats: `prior_games = df[(team) & ((season < s) | (season == s & week < w))]`

## API Integrations
| Source | Purpose | Notes |
|--------|---------|-------|
| nfl_data_py | Schedule, play-by-play | Local, no key needed |
| ESPN scores | Completed game scores | Runtime, public API |
| SMTP email | Bet notifications | `emailer.py`, Gmail App Passwords |

No runtime API calls except ESPN scores for completed games.

## Key Components
- `build_and_train_pipeline.py` — runs both pipeline steps (~5 min)
- `nfl-gather-data.py` — feature engineering + XGBoost training
- `create-nfl-historical.py` — historical data fetch via nfl_data_py
- `player_props/train_models.py` — player prop model training
- `scripts/export_best_bets.py` — `best_bets_today.json` writer
- `scripts/send_rich_email_now.py` — SMTP email sender
- `scripts/generate_rss.py` — `alerts_feed.xml` RSS feed

## Storage
All data in `data_files/` (committed to git):
- `nfl_games_historical_with_predictions.csv` — main dataset + predictions
- `model_feature_importances.csv`, `model_metrics.json` — model eval
- `best_bets_today.json` — Sports Picks Grid feed
- `data_files/exports/` — PDF exports

## Memory Optimisation (Streamlit Cloud)
- All numeric columns → `float32` (50% reduction vs `float64`)
- Use DataFrame views not `.copy()`
- All data loading via `@st.cache_data` — NEVER at module level (causes silent crashes)
