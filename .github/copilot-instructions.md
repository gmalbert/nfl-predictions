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
- **Memory optimization**: For Streamlit Cloud deployment, implement memory-efficient patterns:
  - Use `float32` dtypes for numeric columns (50% memory reduction vs float64)
  - Use `Int8` for boolean/binary columns instead of float64
  - Create DataFrame views with filtering instead of `.copy()` to avoid memory duplication
  - Use `@st.cache_data(show_spinner=False)` for technical pages to suppress cache messages
  - Implement pagination for large datasets (>10k rows) with user warnings
  - Avoid module-level data loading - always use lazy loading with caching
- **Loading progress**: Always show detailed progress during data loading with `st.progress()` and descriptive text updates. Example:
  ```python
  with st.spinner("Loading data..."):
      progress_bar = st.progress(0)
      progress_bar.progress(25, text="Loading step 1...")
      # load data
      progress_bar.progress(50, text="Loading step 2...")
      # load more data
      progress_bar.progress(100, text="Ready!")
      time.sleep(0.5)
      progress_bar.empty()
  ```
- **Cache management**: Provide users control over cached data with sidebar settings. Example:
  ```python
  # Add to sidebar
  with st.sidebar:
      st.write("### ⚙️ Settings")
      
      if st.button("🔄 Refresh Data", help="Clear cache and reload all data"):
          st.cache_data.clear()
          st.rerun()
  ```
- **In-app notifications**: Alert users to elite and strong betting opportunities using session state tracking. Example:
  ```python
  # Store new bets in session state to avoid duplicates
  if 'notified_games' not in st.session_state:
      st.session_state.notified_games = set()
  
  # Check for new elite bets (≥65% confidence)
  elite_bets = predictions_df[
      (predictions_df.get('prob_underdogWon', 0) >= 0.65) | 
      (predictions_df.get('prob_underdogCovered', 0) >= 0.65) |
      (predictions_df.get('prob_overHit', 0) >= 0.65)
  ]
  
  new_elite_bets = elite_bets[~elite_bets['game_id'].isin(st.session_state.notified_games)]
  
  if len(new_elite_bets) > 0:
      st.toast(f"🔥 {len(new_elite_bets)} new elite betting opportunities!", icon="🔥")
      st.session_state.notified_games.update(new_elite_bets['game_id'].tolist())
  
  # Check for new strong bets (60-65% confidence)
  strong_bets = predictions_df[
      ((predictions_df.get('prob_underdogWon', 0) >= 0.60) & (predictions_df.get('prob_underdogWon', 0) < 0.65)) | 
      ((predictions_df.get('prob_underdogCovered', 0) >= 0.60) & (predictions_df.get('prob_underdogCovered', 0) < 0.65)) |
      ((predictions_df.get('prob_overHit', 0) >= 0.60) & (predictions_df.get('prob_overHit', 0) < 0.65))
  ]
  
  new_strong_bets = strong_bets[~strong_bets['game_id'].isin(st.session_state.notified_games)]
  
  if len(new_strong_bets) > 0:
      st.toast(f"⭐ {len(new_strong_bets)} new strong betting opportunities!", icon="⭐")
      st.session_state.notified_games.update(new_strong_bets['game_id'].tolist())
  ```
- **Performance dashboard**: Track model accuracy and profitability with betting log analysis. Example:
  ```python
  # Load betting log and calculate metrics
  betting_log = pd.read_csv('data_files/betting_recommendations_log.csv')
  completed_bets = betting_log[betting_log['bet_result'].isin(['win', 'loss'])]
  
  # Overall metrics
  win_rate = (completed_bets['bet_result'] == 'win').mean() * 100
  roi = calculate_roi(betting_log)
  units_won = completed_bets['bet_profit'].sum() / 100
  
  # Performance by confidence tier
  confidence_performance = completed_bets.groupby('confidence_tier').agg({
      'bet_result': lambda x: (x == 'win').mean() * 100,
      'bet_profit': 'sum'
  })
  ```
- **Bankroll management**: Smart position sizing for elite bets with Kelly-inspired risk management. Example:
  ```python
  # Bankroll input and risk calculation
  bankroll = st.number_input("Current Bankroll ($)", value=10000, step=100)
  risk_pct = {"Conservative (1%)": 0.01, "Moderate (2%)": 0.02}[risk_level]
  bet_amount = bankroll * risk_pct
  
  # Elite bet identification (≥65% confidence)
  elite_bets = predictions_df[(predictions_df['prob_underdogWon'] >= 0.65) & 
                              (predictions_df['pred_underdogWon_optimal'] == 1)]
  
  # Expected value calculation
  expected_value = (confidence * payout_multiplier - 100)
  ```
- **Session state**: Use reset flag pattern for filters, not `.clear()`.}
- **Navigation**: Use `st.switch_page()` for page changes.
- **Model training**: Three XGBoost models (spread, moneyline, totals), calibrated with isotonic regression. Betting thresholds are F1-optimized (not 50%).
- **Betting log**: All recommendations tracked in `betting_recommendations_log.csv`. Results auto-fetched from ESPN API.
- **Memory optimization**: Use `float32` for numeric columns, `Int8` for boolean columns, DataFrame views instead of copies, `@st.cache_data(show_spinner=False)` for technical pages to reduce memory usage and suppress cache messages.

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
- **RESOLVED**: Memory resource limits on Streamlit Cloud - implemented dtype optimizations, DataFrame views, and pagination

## References
- See `README.md` for project summary and local setup
- See `ROADMAP.md` for planned features
- See `data_files/` for all model/data artifacts
