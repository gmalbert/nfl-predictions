# Copilot Instructions: NFL Predictions Project

## Overview
Multi-page Streamlit app for NFL betting predictions using XGBoost models. Predicts outcomes for spread, moneyline, over/under markets, and player props (passing/rushing/receiving yards, TDs). All data and models are pre-computed locally; no runtime API calls except ESPN score fetching.

## Architecture & Data Flow
**Data Pipeline (Sequential)**:
1. `create-nfl-historical.py` → Fetches NFL schedule/game data via `nfl_data_py` → `nfl_games_historical.csv`
2. `nfl-gather-data.py` → Feature engineering + XGBoost training + predictions → `nfl_games_historical_with_predictions.csv`
3. Run both via: `python build_and_train_pipeline.py` (~5 min runtime)

**UI Layer**:
- `predictions.py` → Main dashboard (betting tabs, metrics, PDF exports)
- `pages/1_📊_Historical_Data.py` → Advanced filtering over 196k+ play-by-play records
- `pages/2_🎯_Player_Props.py` → Player performance predictions (yards, TDs)
  - **New**: includes a `DK Pick 6 Calculator` tab for entering DraftKings Pick 6 over/under lines and receiving an OVER/UNDER recommendation. The calculator uses cached XGBoost models located in `player_props/models` (loaded via `load_xgb_models()` in the page) and falls back to a Laplace-smoothed historical hit rate when a model/tier is unavailable.
- `pages/3_🎲_Parlay_Builder.py` → Multi-bet parlay construction and analysis
- `pages/4_📈_Model_Performance.py` → Model evaluation and calibration metrics
- All data loaded via `@st.cache_data` decorators (never at module level)
- Feature lists (`best_features_*.txt`) must stay synchronized between training and UI

**Critical Constraint**: All features must be pre-game only (zero data leakage). Rolling stats exclude current game.

## Critical Conventions

### Memory Optimization (Streamlit Cloud)
```python
# Dtype pattern - apply to all DataFrames
df['numeric_col'] = df['numeric_col'].astype('float32')  # 50% memory reduction
df['boolean_col'] = df['boolean_col'].astype('Int8')     # vs float64
# Use DataFrame views, not .copy(), to avoid duplication
filtered_df = df[df['season'] == 2025]  # Good: creates view
```

### Lazy Data Loading (MANDATORY)
```python
# WRONG - causes silent crashes on Streamlit Cloud
predictions_df = pd.read_csv('data_files/predictions.csv')  # Module-level = BAD

# CORRECT - use caching
@st.cache_data
def load_predictions_csv():
    return pd.read_csv('data_files/predictions.csv', sep='\t')

# In function/page context only
predictions_df = load_predictions_csv()
```

### Model Training - Spread Prediction Inversion
**CRITICAL BUG FIX (Dec 2025)**: Spread model predictions are inverted due to `underdogCovered` target definition:
```python
# In nfl-gather-data.py, after spread model predictions
prob_underdogCovered = 1 - prob_underdogCovered  # Fix inversion
# Impact: ROI improved from -90% to +60%, win rate 3.6% → 91.9%
```

### Betting Thresholds
- **Spread**: 52.4% breakeven (EV-based for -110 odds), warnings below 45%
- **Moneyline**: 28% (F1-optimized for underdogs)
- **Totals**: F1-optimized thresholds
- Confidence tiers: Elite (≥65%), Strong (60-65%), Good (55-60%), Standard (50-55%)

### UI Patterns
```python
# HTML Download Buttons with embedded icons
with open('data_files/pdf_icon.png', 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode('ascii')
img_tag = f'<img src="data:image/png;base64,{img_b64}" style="width:36px;height:36px;...">'
html = f'<a download="{filename}" href="{data_uri}">{img_tag}<span>Download</span></a>'
st.markdown(html, unsafe_allow_html=True)

# Session state for notifications (avoid duplicates)
if 'notified_games' not in st.session_state:
    st.session_state.notified_games = set()
if game_id not in st.session_state.notified_games:
    st.toast("🔥 New elite bet!", icon="🔥")
    st.session_state.notified_games.add(game_id)

# Tab structure
tab1, tab2 = st.tabs(["Spread Bets", "Moneyline Bets"])
with tab1:
    if predictions_df is not None:
        st.dataframe(predictions_df, use_container_width=True)

# In-app pipeline trigger
```python
# The Upcoming Games expander shows a "🔄 Generate Predictions" button
# when scheduled games lack model outputs. The button runs the local
# pipeline (`build_and_train_pipeline.py`) and refreshes the UI on success.
if tbd_count > 0:
  if st.button("🔄 Generate Predictions"):
    subprocess.run(["python", "build_and_train_pipeline.py"])  # run locally
```
```

### PDF Export Pattern
```python
# Generate on-demand only (not at module load)
def generate_pdf_bytes(df_upcoming) -> bytes:
    buffer = BytesIO()
    # Use ReportLab, landscape letter, compact headers
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter))
    # Filter to upcoming games only, save to data_files/exports/
    return buffer.getvalue()
```

## Developer Workflow
- **Run app**: `streamlit run predictions.py`
- **Generate predictions / Build & Train (single-step)**: `python build_and_train_pipeline.py` (takes ~5 min)
  - To run training only (features + models): `python nfl-gather-data.py`
- **Python version**: Must use 3.12
- **Local testing**: Activate venv, run above commands
- **Deployment**: Streamlit Cloud, all data files committed
- **Performance**: `.streamlit/config.toml` increases timeouts/message size for large data

### Developer Scripts & Checks
- When creating new helper or check scripts (for model diagnostics, calibration checks, or data validation), create them as Python files and place them in the `scripts/` folder (e.g., `scripts/check_moneyline_calibration.py`, `scripts/analyze_underdog_impact.py`).
- All new scripts must follow the project's lazy-loading and dtype guidelines and be import-safe (should not perform heavy data loads at module import time).
- Add a one-line description at the top of each script and include a simple `if __name__ == '__main__':` runner so they can be executed directly by CI or from the command line.


## Patterns & Examples
- **Adding features**: Update feature lists in both `nfl-gather-data.py` and `predictions.py`, validate no leakage, retrain models, update best features files
- **Adding tabs**:
  ```python
  with tab_name:
      st.write("### Section Title")
      if predictions_df is not None:
          st.dataframe(predictions_df[...])
      else:
          st.warning("Data not available")
  ```
- **Player Props**: Load player stats from `data_files/player_*.csv`, use XGBoost models for yards/TD predictions, display in `pages/2_🎯_Player_Props.py`
 - **DK Pick 6 Calculator**: `pages/2_🎯_Player_Props.py` now contains an interactive calculator where users can:
   - Search/select a player, choose stat category (auto-suggested by position), and enter the DraftKings Pick 6 line.
   - See both the ML-model probability (when available) and a historical hit-rate fallback (Laplace smoothing).
   - View the prediction source in the UI: `🤖 ML Model` (model chosen by season-average tier) or `📊 Historical` (fallback).
   - Models and feature logic live under `player_props/predict.py` and model files are JSONs in `player_props/models/`.

  Developer notes:
  - Models are loaded with `@st.cache_data` to avoid repeated heavy loads.
  - Feature extraction uses L3/L5/L10 rolling stats plus auxiliary features (TDs, attempts, completions, targets) and basic matchup defaults (`opponent_def_rank`, `is_home`, `days_rest`).
  - If you need to regenerate player prop models, run `python player_props/predict.py` per the pipeline.
- **Parlay Builder**: In `pages/3_🎲_Parlay_Builder.py`, combine bets from predictions_df, calculate parlay odds = product of individual probabilities
- **Betting logic**: Confidence tiers—Elite (≥65%), Strong (60-65%), Good (55-60%), Standard (50-55%).

## Integration Points
- **External data**: All historical/play-by-play data is pre-fetched and stored in `data_files/`. No runtime API calls except ESPN scores for completed games.
- **Feature importances/metrics**: Stored in `model_feature_importances.csv` and `model_metrics.json`.
- **Automated workflows**: GitHub Actions run nightly updates during NFL season (Sept-Feb), including ESPN scores, smart play-by-play data updates, and model predictions. Weekly email predictions sent Wednesdays at 8 PM ET.
- **Email notifications**: Enhanced HTML emails with clear betting recommendations:
  - Format: "**TEN +2.5** to cover (69.1%) 🔥 ELITE" instead of cryptic probabilities
  - Individual confidence badges per bet (ELITE ≥65%, STRONG 60-65%, GOOD 55-60%)
  - Smart filtering (Spread ≥50%, Moneyline ≥28%, Totals ≥50%)
  - Preview: `python scripts/preview_email.py`
  - Send: `python scripts/send_rich_email_now.py`
  - Uses SMTP via `emailer.py` (Gmail App Passwords supported)
- **RSS feed**: `scripts/generate_rss.py` generates `alerts_feed.xml` with per-game links using base URL from `app_config.json`.

## Known Issues
- Python 3.13 not supported
- Module-level data loading causes silent crashes—always use lazy pattern
- Large CSVs: Use tab-separated, cache with `@st.cache_data`
- **RESOLVED**: Memory resource limits on Streamlit Cloud - implemented dtype optimizations, DataFrame views, and pagination

## References
- See `README.md` for project summary and local setup
- See `ROADMAP.md` for planned features
- See `data_files/` for all model/data artifacts

## Recent Changes (Dec 13, 2025)

### Critical Model Fix: Prediction Inversion 🎉
- **Issue**: Spread model predictions were inverted (low confidence = high accuracy, high confidence = low accuracy)
- **Root Cause**: Target variable `underdogCovered` definition had incorrect logic
- **Fix**: Applied `prob_underdogCovered = 1 - prob_underdogCovered` in `nfl-gather-data.py` after predictions
- **Impact**: 
  - ROI: -90% → +60%
  - Profitable games: 0 → 62 (out of 63 remaining)
  - Max confidence: 48% → 89.5%
  - Calibration error: 45% → 28%

### New Features Added
- **Momentum features** (8): Last 3 games win%, scoring averages, point differential trends
- **Rest advantage features** (5): Rest day differences, well-rested/short-rest flags (≥10 days, ≤6 days)
- **Weather impact features** (3): Cold weather (≤32°F), windy (≥15mph), extreme conditions
- **Total new features**: 18 (calculated without data leakage using rolling windows)

### UI Improvements
- **EV explanation**: Added expandable section explaining Expected Value concept
- **Spread bet sorting**: Changed to date ascending (earliest games first) instead of confidence descending
- **Fixed Unicode issues**: Removed emoji characters causing encoding errors in training output

### Documentation
- Created `MODEL_FIX_PLAN.md`: Comprehensive analysis of calibration issues and solutions
- Created `NEW_FEATURES_DEC13.md`: Details on momentum/rest/weather features
- Updated `SPREAD_THRESHOLD_CHANGE.md`: Documents threshold evolution (50% → 52.4% → 45% with EV warnings)

### Technical Notes
- Model is now **profitable and functional** - optional improvements (hyperparameter tuning, ensemble methods) documented for future enhancement
- Momentum features will become more impactful as 2025 season progresses (currently early-season with limited data)
- All analysis scripts preserved in repo: `analyze_model_issues.py`, `test_inversion_fix.py`, `check_remaining_games.py`
