> **AI Onboarding Guide** — See also `.github/copilot-instructions.md` for full coding conventions.

# NFL Predictions — Site Summary

## What This App Does

Streamlit multi-page app for NFL betting predictions. Uses XGBoost models to predict spread, moneyline, and over/under outcomes plus player props (passing/rushing/receiving yards, TDs). Includes a DraftKings Pick 6 calculator, parlay builder, and model performance tracking. A two-step pipeline generates predictions before the UI is run.

## Quick Start

```bash
# 1. Activate virtual environment
.\.venv\Scripts\Activate.ps1        # Windows
source .venv/bin/activate           # macOS/Linux

# 2. Run the data + training pipeline (~5 minutes)
python build_and_train_pipeline.py

# 3. Run the app
streamlit run predictions.py
```

**Important**: Step 2 must complete before Step 3. The UI reads pre-generated prediction CSVs.

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit (multi-page: main + 4 pages) |
| ML | XGBoost (spread, moneyline, totals, player props) |
| Data source | `nfl_data_py` (schedule, game data, play-by-play) |
| PDF export | ReportLab |
| Python | 3.12 (3.13 not supported) |

## Key Files

| File | Purpose |
|---|---|
| `predictions.py` | Main dashboard — betting tabs, metrics, PDF exports |
| `pages/1_Historical_Data.py` | Advanced filtering over 196k+ play-by-play records |
| `pages/2_Player_Props.py` | Player performance predictions + DK Pick 6 Calculator |
| `pages/3_Parlay_Builder.py` | Multi-bet parlay construction and analysis |
| `pages/4_Model_Performance.py` | Model evaluation, calibration metrics |
| `create-nfl-historical.py` | Step 1: fetch NFL schedule/game data → `nfl_games_historical.csv` |
| `nfl-gather-data.py` | Step 2: feature engineering + XGBoost training + predictions |
| `build_and_train_pipeline.py` | Runs both steps sequentially |
| `player_props/` | Player prop models (XGBoost + LightGBM soft-voting ensembles) |
| `best_features_*.txt` | Feature lists — must stay in sync between training and UI |

## Data Flow

1. `create-nfl-historical.py` → `nfl_data_py` fetches schedule/game data → `nfl_games_historical.csv`
2. `nfl-gather-data.py` → feature engineering (70+ features including momentum, rest, weather) → XGBoost training → `nfl_games_historical_with_predictions.csv`
3. Streamlit UI reads prediction CSV via `@st.cache_data` — never at module level
4. Live ESPN scores fetched at runtime for completed game badge updates

## Critical Bug Fix (December 2025)

**Spread model predictions are inverted** — this is a known issue that was fixed:
```python
# In nfl-gather-data.py, after spread model predictions
prob_underdogCovered = 1 - prob_underdogCovered  # MUST keep this line
```
Without this inversion, ROI drops from +60% to -90%. Never remove it.

## Betting Thresholds

| Market | Threshold | Notes |
|---|---|---|
| Spread | 50% | Natural binary decision boundary |
| Moneyline | 28% | F1-optimized for underdog detection |
| Totals | F1-optimized | See model metrics |

## Environment Variables

No external API keys required. All historical data is pre-fetched via `nfl_data_py`. ESPN live scores endpoint is public.

## Critical Conventions

- **Never** load data at module level — always use `@st.cache_data` (silent crash on Streamlit Cloud)
- Python 3.12 required — 3.13 is not supported
- `best_features_*.txt` files must stay synchronized between `nfl-gather-data.py` and `predictions.py`
- All dtype optimizations must be applied after CSV loads: `df['float_col'].astype('float32')` reduces memory 50%
- Use `width='stretch'` for dataframes/charts — `use_container_width` is removed

## Common Gotchas

- `build_and_train_pipeline.py` takes ~5 minutes — run it before the UI, not during
- Player prop models live in `player_props/` and are separate from the game outcome models
- New features must be added in **both** `nfl-gather-data.py` and `predictions.py`; the feature list files must also be updated
- The DK Pick 6 calculator uses Laplace-smoothed historical hit rate as fallback when a model tier is unavailable
