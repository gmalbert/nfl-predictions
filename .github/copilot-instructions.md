# GitHub Copilot Instructions for NFL Predictions Project

## Project Overview
This is a Streamlit-based NFL betting predictions application that uses machine learning (XGBoost) to predict game outcomes and provide betting recommendations for three markets:
- **Spread betting** (will underdog cover the spread?)
- **Moneyline betting** (will underdog win outright?)
- **Over/Under betting** (will total score go over the line?)

## Tech Stack
- **Frontend**: Streamlit 1.51.0
- **ML Framework**: XGBoost with scikit-learn
- **Data Processing**: pandas, numpy
- **Data Source**: nflverse/nfl-data-py for historical play-by-play data
- **Python Version**: 3.12 (required for Streamlit Cloud compatibility)

## Key Files
- `predictions.py` - Main Streamlit application (2000+ lines)
- `nfl-gather-data.py` - Data collection and feature engineering script
- `create-nfl-historical.py` - Historical data processing
- `data_files/` - All data, models, and feature files
  - `nfl_games_historical_with_predictions.csv` - Main predictions dataset
  - `best_features_*.txt` - Feature lists for each model
  - `model_metrics.json` - Model performance metrics
  - `betting_recommendations_log.csv` - Betting history tracking

## Architecture Principles

### Data Loading
- **Always use lazy loading** - Never load data at module level
- Use `@st.cache_data` decorator for all data loading functions
- Load data only when the UI is accessed, not during import
- Example pattern:
  ```python
  @st.cache_data
  def load_data():
      return pd.read_csv(path, sep='\t')
  
  # At module level
  data = None
  
  # Later, when needed
  if data is None:
      data = load_data()
  ```

### Feature Engineering
- All features must be **pre-game only** (no data leakage from actual game results)
- Rolling statistics are calculated from previous games only
- Feature lists are maintained in sync between `nfl-gather-data.py` and `predictions.py`
- Best features are stored in text files and loaded dynamically

### Model Training
- Three separate XGBoost models (spread, moneyline, totals)
- Models use calibrated probabilities (isotonic regression)
- Optimal betting thresholds are determined by F1-score optimization (not 50%)
- Monte Carlo feature selection available in the UI

### Streamlit Best Practices
- `st.set_page_config()` must be the FIRST Streamlit command
- Use tabs for organizing complex UI sections
- Use expanders for collapsible content
- Use `st.cache_data` for expensive operations
- Use column layouts for metrics and comparisons

## Code Style
- Use descriptive variable names (e.g., `predictions_df`, `historical_game_level_data`)
- Keep functions focused and single-purpose
- Add docstrings to all cached functions
- Use f-strings for string formatting
- Prefer explicit column selection over inferring data types

## Common Patterns

### Adding a New Tab
```python
with tab_name:
    st.write("### Section Title")
    
    if predictions_df is not None:
        # Process and display data
        filtered_df = predictions_df[predictions_df['condition']]
        
        st.dataframe(
            filtered_df,
            column_config={
                'column': st.column_config.NumberColumn('Label', format='%.2f')
            }
        )
    else:
        st.warning("Data not available")
```

### Adding a New Feature
1. Update feature list in `nfl-gather-data.py`
2. Calculate feature in data pipeline
3. Update feature list in `predictions.py`
4. Retrain models and save new best features

### Betting Logic
- Betting recommendations require optimal threshold to be met
- Confidence tiers: Elite (>65%), Strong (60-65%), Good (55-60%), Standard (50-55%)
- All bets are tracked in `betting_recommendations_log.csv`
- Results are automatically fetched from ESPN API

## Deployment Notes
- Deployed on Streamlit Cloud
- Requires Python 3.12 (specified in `.python-version`)
- All data files are committed to git (no external data sources needed at runtime)
- Environment variables: None required

## Known Issues & Solutions
- **Python 3.13 incompatibility**: Always use Python 3.12
- **Module-level data loading**: Causes silent crashes - use lazy loading pattern
- **Large CSV files**: Use `sep='\t'` for tab-separated files, cache with `@st.cache_data`
- **Streamlit Cloud timeouts**: Ensure all heavy operations are cached

## Testing Locally
```bash
# Activate virtual environment
.\venv\Scripts\Activate

# Run the app
streamlit run predictions.py

# Generate fresh predictions (takes ~5 minutes)
python nfl-gather-data.py
```

## When Modifying Code
1. **Check Python version compatibility** - Must work with Python 3.12
2. **Test data loading** - Ensure lazy loading pattern is maintained
3. **Verify caching** - All expensive operations should use `@st.cache_data`
4. **Update feature lists** - Keep `nfl-gather-data.py` and `predictions.py` in sync
5. **Test on Streamlit Cloud** - Local testing may not catch deployment issues

## Useful Contexts
- **Betting math**: All probabilities are calibrated, thresholds are F1-optimized
- **Data freshness**: Historical data goes through 2024 season
- **Model retraining**: Required when adding features or changing target variables
- **UI performance**: Use expandable sections for large data displays to improve initial load time
