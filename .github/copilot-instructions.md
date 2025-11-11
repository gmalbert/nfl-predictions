# Copilot Instructions: NFL Predictions Project

## Overview
Multi-page Streamlit app for NFL betting predictions using XGBoost models. Predicts outcomes for spread, moneyline, and over/under markets. All data and models are local; no runtime API calls required.

## Architecture & Data Flow
- Main UI: `predictions.py` (core logic, tabs, betting analysis)
- Historical Data: `pages/1_📊_Historical_Data.py` (advanced filtering, 196k+ rows)
- Data/feature engineering: `nfl-gather-data.py`, `create-nfl-historical.py`
- Data files: `data_files/` (CSV, model metrics, feature lists)
- All features must be pre-game only (no data leakage)
- Feature lists (`best_features_*.txt`) must be kept in sync between data pipeline and UI

## Key Conventions
- **Lazy data loading**: Never load data at module level. Use `@st.cache_data` for all data loading functions. Example:
  ```python
  @st.cache_data
  def load_predictions_csv():
      return pd.read_csv(path, sep='\t')
  predictions_df = None
  if predictions_df is None:
      predictions_df = load_predictions_csv()
  ```
- **No data leakage**: All features/statistics must use only pre-game info. Rolling stats must exclude current game.
- **UI structure**: Use tabs for sections, expanders for large data, columns for metrics. Use `width='stretch'` for dataframes.
- **Session state**: Use reset flag pattern for filters, not `.clear()`.
- **Navigation**: Use `st.switch_page()` for page changes.
- **Model training**: Three XGBoost models (spread, moneyline, totals), calibrated with isotonic regression. Betting thresholds are F1-optimized (not 50%).
- **Betting log**: All recommendations tracked in `betting_recommendations_log.csv`. Results auto-fetched from ESPN API.

## Developer Workflow
- **Run app**: `streamlit run predictions.py`
- **Generate predictions**: `python nfl-gather-data.py` (takes ~5 min)
- **Python version**: Must use 3.12
- **Local testing**: Activate venv, run above commands
- **Deployment**: Streamlit Cloud, all data files committed
- **Performance**: `.streamlit/config.toml` increases timeouts/message size for large data

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
- **Betting logic**: Confidence tiers—Elite (≥65%), Strong (60-65%), Good (55-60%), Standard (50-55%).

## Integration Points
- **External data**: All historical/play-by-play data is pre-fetched and stored in `data_files/`. No runtime API calls except ESPN scores for completed games.
- **Feature importances/metrics**: Stored in `model_feature_importances.csv` and `model_metrics.json`.

## Known Issues
- Python 3.13 not supported
- Module-level data loading causes silent crashes—always use lazy pattern
- Large CSVs: Use tab-separated, cache with `@st.cache_data`

## References
- See `README.md` for project summary and local setup
- See `ROADMAP.md` for planned features
- See `data_files/` for all model/data artifacts
