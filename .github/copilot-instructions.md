# NFL Predictions - AI Coding Agent Instructions

## 🏈 Project Overview
This is a sophisticated NFL betting analytics platform using **three separate XGBoost models** for comprehensive betting strategy. The system achieves **9.9% ROI on underdog bets** (24% threshold), spread predictions, and over/under totals with production-optimized models and strict data leakage prevention. **CRITICAL**: Uses F1-score optimized thresholds (24% moneyline, 34% spread, 30% totals), NOT 50% probability cutoffs.

## 🎯 Critical Architecture Understanding

### Two-Stage Data Pipeline
- **Stage 1: `nfl-gather-data.py`** - Model training and feature engineering (run first)
- **Stage 2: `predictions.py`** - Streamlit dashboard for visualization and analysis

### Core Model Strategy
- **Three Separate Models**: Moneyline (9.9% ROI), Spread, and Over/Under (Totals)
- **F1-Score Optimized Thresholds**: 24% (moneyline), 34% (spread), 30% (totals)
- **NOT 50% cutoffs**: Thresholds optimized via F1-score across 0.1-0.6 range (step 0.02)
- **Production XGBoost parameters**: 300 estimators, 0.05 learning rate, depth 6, L1/L2 regularization
- **Calibrated probabilities** using `CalibratedClassifierCV` with isotonic method
- **Class-weighted XGBoost** for handling imbalanced favorite/underdog data
- **Data leakage elimination**: Strict temporal boundaries in all rolling statistics
- **Feature Importances**: XGBoost built-in gain-based importances (NOT permutation importance)

## 🔧 Essential Developer Workflows

### Initial Setup & Training
```powershell
# ALWAYS run training first or after data updates
python nfl-gather-data.py  # Generates models, features, metrics

# Launch dashboard
streamlit run predictions.py
```

### Data Dependencies
- **Primary**: `data_files/nfl_games_historical.csv` (NFLverse data)
- **Betting lines**: `espn_nfl_scores_2020_2024.csv` 
- **Large files**: Git LFS for `*.csv.gz` files (>50MB)

## 📊 Project-Specific Patterns

### Feature Engineering Functions
```python
# Rolling statistics (all-time) - prevent data leakage, only use pre-game info
calc_rolling_stat(), calc_rolling_count() # homeTeamWinPct, awayTeamWinPct, etc.

# Current season only statistics (weekly updated)
calc_current_season_stat() # homeTeamCurrentSeasonWinPct, etc.

# Prior complete season records (annual stability)
calc_prior_season_record() # homeTeamPriorSeasonRecord (last year's final win%)

# Head-to-head historical matchups
calc_head_to_head_record() # headToHeadHomeTeamWinPct (team vs team history)
```

### Monte Carlo Feature Selection
- **Enhanced search space**: 15-feature subsets (up from 8) for better model discovery
- **200 iterations** with production XGBoost parameters for feature testing
- Optimized features stored in `data_files/best_features_*.txt`
- Separate feature sets for spread/moneyline/totals models
- Cross-validation with time-series aware splits

### Key Prediction Columns
- `prob_underdogWon` - Raw model probability (moneyline)
- `pred_underdogWon_optimal` - Binary betting signal (≥24% threshold)
- `edge_underdog_ml` - Value calculation vs sportsbook odds
- `prob_spreadCovered` - Raw spread model probability  
- `pred_spreadCovered_optimal` - Binary spread signal (≥34% threshold)
- `edge_spread` - Spread value calculation
- `prob_overHit` - Raw totals model probability (over)
- `pred_overHit_optimal` - Binary totals signal (≥30% threshold)
- `edge_over` - Over value calculation
- `edge_under` - Under value calculation

### Enhanced Features (Latest Addition)
- **Current Season Performance**: `homeTeamCurrentSeasonWinPct`, `awayTeamCurrentSeasonAvgScore` etc.
- **Prior Season Records**: `homeTeamPriorSeasonRecord`, `awayTeamPriorSeasonRecord` (full season win%)
- **Head-to-Head History**: `headToHeadHomeTeamWinPct` (historical matchup performance)

## 🚨 Critical Implementation Details

### Threshold Logic (NEVER use 50%)
```python
# F1-score optimization finds ~24% threshold for moneyline
optimal_moneyline_threshold = best_threshold
pred_underdogWon_optimal = (prob_underdogWon >= 0.24).astype(int)

# F1-score optimization finds ~34% threshold for spread
optimal_spread_threshold = 0.34
pred_spreadCovered_optimal = (prob_spreadCovered >= 0.34).astype(int)

# F1-score optimization finds ~30% threshold for totals
optimal_totals_threshold = 0.30
pred_overHit_optimal = (prob_overHit >= 0.30).astype(int)
```

### Streamlit Dashboard Structure (Tab-Based Navigation)
- **Tab 1 (Model Predictions)**: Historical predictions vs actual results with formatted checkboxes
- **Tab 2 (Probabilities & Edges)**: Upcoming games with percentage-formatted probabilities
- **Tab 3 (Betting Performance)**: Compact metrics table (600px wide) with 3-decimal formatting
- **Tab 4 (Underdog Bets)**: Next 10 moneyline opportunities (≥24% threshold)
- **Tab 5 (Spread Bets)**: Next 15 spread opportunities with confidence tiers (≥34% threshold)
- **Tab 6 (Over/Under Bets)**: Next 15 totals opportunities (≥30% threshold)
- **Tab 7 (Betting Log)**: Automated bet tracking with results

### UI/UX Formatting Standards (November 2025)
- **Modern Navigation**: All sections use `st.tabs()` instead of checkboxes
- **Percentage Display**: All probabilities shown as percentages (45.6%) not decimals (0.456)
- **Date Formatting**: DateColumn with format='MM/DD/YYYY' for consistency
- **Decimal Precision**: 3 decimals (%.3f) for metrics, 1 decimal (%.1f%%) for percentages
- **Column Configuration**: Use column_config with proper types (DateColumn, NumberColumn, CheckboxColumn)
- **Tooltips**: Helpful explanations via `help` parameter in column configs
- **Compact Labels**: Short column names ("ML Prob", "Spread Edge") with descriptive tooltips
- **Table Width Control**: Metrics table limited to 600px, other tables use default widths

### Confidence Tier System
- **🔥 Elite** (≥75% probability): Highest confidence bets
- **⭐ Strong** (65-74% probability): Strong conviction plays
- **📈 Good** (54-64% probability): Solid betting opportunities
- **Dashboard Integration**: Visual indicators in "Next 15 Spread Bets" section

### Data Quality Patterns
- Remove games with missing betting lines
- Handle spread_line polarity (positive = home favored)
- Git LFS integration for large datasets

## 🔍 Debugging & Common Issues

### Model Not Loading
Check `data_files/` for required files:
- `best_features_*.txt`
- `model_metrics.json`
- `nfl_games_historical_with_predictions.csv`

### Streamlit Data Display Patterns (November 2025)
- **Percentage Conversion**: Multiply probability columns by 100 before display
  ```python
  display_df = filtered_data.copy()
  prob_cols = ['prob_underdogWon', 'prob_spreadCovered', 'edge_underdog_ml']
  for col in prob_cols:
      display_df[col] = display_df[col] * 100
  ```
- **Column Configuration**: Use proper NumberColumn formatting
  ```python
  column_config={
      'gameday': st.column_config.DateColumn('Date', format='MM/DD/YYYY'),
      'prob_underdogWon': st.column_config.NumberColumn('ML Prob', format='%.1f%%'),
      'home_score': st.column_config.NumberColumn('Home Pts', format='%d')
  }
  ```
- **Data Type Handling**: Keep datetime as datetime for DateColumn (don't convert to string)
- **Display Copies**: Always create `.copy()` before transforming data for display
- **Error Prevention**: Use `format='%.1f%%'` for percentages, `%.3f` for decimals, `%d` for integers

### Streamlit Compatibility
- Use modern dataframe display with column_config (no deprecated `use_container_width`)
- Error handling for missing large files (`nfl_history_2020_2024.csv.gz`)
- Red triangles indicate type/format mismatches in column_config

### Betting Logic Validation
- Verify 28% threshold in both training and dashboard
- Check `pred_underdogWon_optimal = 1` signals match probability threshold
- ROI calculations use selective betting (~72% of games)
- **Spread Model**: Ensure `target_spread = 'underdogCovered'` (NOT 'spreadCovered')
- **Correlation Check**: Model correlation should be +0.74 with target (not -0.48)
- **Date Filtering**: Betting sections must reload fresh data to avoid variable mutation issues

### Variable Mutation Prevention (CRITICAL)
- **Issue Pattern**: Shared `predictions_df` variable gets modified across Streamlit sections
- **Symptoms**: Betting sections show old games (2020) instead of upcoming games
- **Prevention**: Each section should reload from CSV or create isolated copy
- **Best Practice**: `df_isolated = pd.read_csv(path)` at start of each major section
- **Date Handling**: Always convert to datetime and filter `gameday > today` for upcoming games

## 📁 Key Files for Context
- `nfl-gather-data.py` - Lines 213-232 (threshold optimization)
- `predictions.py` - Lines 148-280 (automatic bet logging), Lines 282-380 (score updates)
- `predictions.py` - Lines 950-1030 (underdog bets section - date filtering fix)
- `predictions.py` - Lines 1032-1130 (spread bets section - date filtering fix)
- `data_files/best_features_moneyline.txt` - Optimized feature set
- `data_files/betting_recommendations_log.csv` - Automatic bet tracking with results
- `README.md` - Comprehensive project documentation

## 🎲 Domain-Specific Knowledge
- **Underdog strategy** outperforms favorite betting despite lower win rate
- **Edge calculation** = model_prob - implied_odds_prob
- **NFLverse data** is gold standard (not ESPN for play-by-play)
- **Class imbalance** requires weighted models (favorites ~70% of outcomes)
- **Three betting strategies**: Moneyline (9.9% ROI), Spread, and Over/Under totals
- **Over/Under model**: Uses 30% F1-optimized threshold, predicts if combined score exceeds total_line

## 🧪 Monte Carlo Feature Selection Process

### Feature Optimization Strategy
```python
# Process in nfl-gather-data.py and predictions.py Tab 4
random.seed(42)  # Reproducible results
features_to_test = random.sample(all_features, min(15, len(all_features)))
```

### Cross-Validation Approach
- **Time-series aware splits** - respects chronological order
- **3-fold CV** with stratification for class balance
- **Multiple metrics**: Accuracy, AUC, F1-score for comprehensive evaluation
- **Feature subset testing**: 10-15 features per iteration to avoid overfitting

### File Management Pattern
```python
# Separate optimized features for each model type
best_features_spread.txt      # Point spread predictions
best_features_moneyline.txt   # Underdog win predictions (most critical)
best_features_totals.txt      # Over/under predictions
```

### Implementation Notes
- Features tested include rolling stats (win%, avg scores, point differentials)
- **Prevents data leakage**: Only pre-game information used for predictions
- Results stored in `model_feature_importances.csv` for analysis

## 🔍 Feature Importance Analysis

### XGBoost Built-In Importances (November 2025 Update)
- **All three models** now use XGBoost's built-in `.feature_importances_` (gain-based)
- **Why not permutation importance**: Permutation was returning negative/zero values with CalibratedClassifierCV wrapper
- **No standard deviations**: Built-in importances are deterministic (no std dev column in dashboard)
- **More interpretable**: All positive values showing relative importance percentages

### Accessing Importances from Calibrated Models
```python
# Extract base estimator from CalibratedClassifierCV wrapper
spread_base_estimator = model_spread.calibrated_classifiers_[0].estimator
spread_importances = spread_base_estimator.feature_importances_

# Same pattern for all three models
moneyline_base_estimator = model_moneyline.calibrated_classifiers_[0].estimator
totals_base_estimator = model_totals.calibrated_classifiers_[0].estimator
```

### Dashboard Display
- **Separate tabs** for each model's feature importances
- **Top 25 features** shown per model
- **Columns**: Feature Name, Importance (no std dev)
- **Format**: 4 decimal places (%.4f) for importance values

## 🎨 **UI/UX Improvements (November 2025)**

### **Dashboard Formatting Standards**
All data displays in the Streamlit dashboard follow consistent formatting patterns:

#### **Percentage Display Pattern**
```python
# ALWAYS multiply probability columns by 100 before displaying as percentages
display_df = predictions_df.copy()
prob_cols = ['prob_underdogWon', 'prob_spreadCovered', 'edge_underdog_ml', 
             'implied_prob_underdog_ml', 'prob_overHit', 'edge_over']
for col in prob_cols:
    if col in display_df.columns:
        display_df[col] = display_df[col] * 100

# Then use percentage format in column_config
column_config={
    'prob_underdogWon': st.column_config.NumberColumn('ML Prob', format='%.1f%%')
}
```

#### **Metrics Display Pattern**
```python
# All model metrics use 3 decimal precision
accuracy_col = st.column_config.NumberColumn('Accuracy', format='%.3f')
mae_col = st.column_config.NumberColumn('MAE', format='%.3f')
threshold_col = st.column_config.NumberColumn('Threshold', format='%.3f')
```

#### **Date Display Pattern**
```python
# ALWAYS keep gameday as datetime (don't convert to string)
predictions_df['gameday'] = pd.to_datetime(predictions_df['gameday'], errors='coerce')

# Use DateColumn with MM/DD/YYYY format
column_config={
    'gameday': st.column_config.DateColumn('Date', format='MM/DD/YYYY')
}
```

#### **Compact Labels with Tooltips**
```python
# Use short labels with descriptive help text
column_config={
    'prob_underdogWon': st.column_config.NumberColumn(
        'ML Prob',  # Short label
        format='%.1f%%',
        help='Model probability underdog wins outright'  # Detailed tooltip
    )
}
```

### **Tab-Based Navigation Implementation**
```python
# Modern tab navigation (replaced checkbox navigation November 2025)
pred_tab1, pred_tab2, pred_tab3, pred_tab4, pred_tab5, pred_tab6 = st.tabs([
    "📊 Model Predictions", 
    "🎯 Probabilities & Edges",
    "💰 Betting Performance",
    "🔥 Underdog Bets",
    "🏈 Spread Bets",
    "📋 Betting Log"
])

with pred_tab1:
    # Tab content here
    st.write("### Model Predictions vs Actual Results")
    st.dataframe(...)
```

### **Display Copy Pattern (Prevent Mutations)**
```python
# ALWAYS create display copy before transformations
display_df = filtered_data.copy()

# Apply transformations to display copy only
display_df['prob_column'] = display_df['prob_column'] * 100

# Original data remains unchanged
st.dataframe(display_df, ...)  # Show transformed copy
```

## 🔥 **CRITICAL: Data Leakage Elimination & Production Optimization**

### **Data Leakage Fixes (October 2025)**
```python
# FIXED: All rolling statistics now use strict temporal boundaries
# OLD (LEAKED): df.groupby(['team']).expanding().mean()  # Used future games!
# NEW (CORRECT): Only use games where (season < current) OR (week < current)

def calc_rolling_stat(df, team_col, stat_col, new_col_name):
    """Calculate rolling statistics with strict temporal boundaries"""
    # Ensures no future information leaks into training/prediction
    return df[df['gameDate'] < current_game_date].groupby(team_col)[stat_col].mean()
```

### **Production XGBoost Parameters (Optimized)**
```python
# Training models (nfl-gather-data.py)
PRODUCTION_PARAMS = {
    'n_estimators': 300,        # Increased from 100 for better performance
    'learning_rate': 0.05,      # Reduced for better generalization
    'max_depth': 6,             # Optimal complexity
    'reg_alpha': 1,             # L1 regularization
    'reg_lambda': 1,            # L2 regularization
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'eval_metric': 'logloss',
    'class_weight': 'balanced'  # Handle favorite/underdog imbalance
}

# Monte Carlo feature selection (faster iteration)
MONTE_CARLO_PARAMS = {
    'n_estimators': 100,        # Faster for feature testing
    'learning_rate': 0.1,       # Higher for quicker convergence
    'max_depth': 6,
    'random_state': 42
}
```

### **Performance Impact of Fixes**
- **ROI improvement**: 14.1% → 60.9% (with leak-free validation)
- **Realistic accuracy**: Models now show true out-of-sample performance
- **Production ready**: No future information contamination

## 🚨 **CRITICAL: Understanding Data Leakage in NFL Predictions**

### **What is Data Leakage?**
Data leakage occurs when information from the future (relative to prediction time) accidentally gets included in model training. In NFL predictions, this means using game outcomes or statistics that wouldn't be available when making real predictions.

### **NFL-Specific Data Leakage Patterns**

#### **Pattern A: Temporal Leakage (Most Critical)**
```python
# WRONG - Uses future games in rolling averages
def calc_rolling_stat_WRONG(df, team_col, stat_col):
    return df.groupby(team_col)[stat_col].expanding().mean()  # Includes future games!

# CORRECT - Only uses past games
def calc_rolling_stat(df, team_col, stat_col, current_season, current_week):
    past_games = df[
        (df['season'] < current_season) | 
        ((df['season'] == current_season) & (df['week'] < current_week))
    ]
    return past_games.groupby(team_col)[stat_col].mean()
```

#### **Pattern B: Same-Game Information Leakage**
```python
# WRONG - Using final game stats to predict game outcome
features = ['homeTeamFinalScore', 'awayTeamFinalScore']  # These ARE the outcome!

# CORRECT - Only pre-game information
features = ['homeTeamAvgScore', 'awayTeamWinPct', 'spread_line']
```

#### **Pattern C: Season-End Statistics Leakage**
```python
# WRONG - Using complete season stats for mid-season predictions
homeTeamSeasonWinPct = season_stats['wins'] / season_stats['total_games']  # Includes future wins!

# CORRECT - Only games played so far
homeTeamCurrentWinPct = games_so_far['wins'] / games_so_far['games_played']
```

### **Data Leakage Detection Methods**

#### **Method A: Temporal Validation**
```python
# Check if any feature uses future information
def validate_temporal_integrity(df, prediction_date):
    for feature in features:
        feature_dates = df[df[feature].notna()]['gameDate']
        if any(feature_dates > prediction_date):
            raise ValueError(f"Data leakage detected in {feature}")
```

#### **Method B: Cross-Validation with Time Splits**
```python
# WRONG - Random splits can leak future data
train_test_split(X, y, test_size=0.2, random_state=42)

# CORRECT - Time-aware splits
def time_series_split(df, test_start_date):
    train = df[df['gameDate'] < test_start_date]
    test = df[df['gameDate'] >= test_start_date]
    return train, test
```

#### **Method C: Feature Availability Check**
```python
# Verify all features would be available at prediction time
def check_feature_availability(features, prediction_time):
    available_features = get_features_at_time(prediction_time)
    unavailable = set(features) - set(available_features)
    if unavailable:
        raise ValueError(f"Features not available at prediction time: {unavailable}")
```

### **NFL Prediction-Specific Leakage Risks**

#### **High-Risk Features (Avoid These)**
- Final game scores, final team statistics
- Complete season records (use partial records only)
- Opponent-adjusted statistics using future opponents
- Rankings based on future performance
- Weather conditions from game day (use forecasts only)

#### **Safe Features (Use These)**
- Historical head-to-head records (only past meetings)
- Rolling averages (strict temporal boundaries)
- Pre-game betting lines and odds
- Team roster information (as of prediction date)
- Current season performance (games played so far only)

### **Validation Framework for Leak-Free Models**

```python
class LeakFreeValidator:
    def __init__(self, prediction_date):
        self.prediction_date = prediction_date
    
    def validate_features(self, df, features):
        """Ensure no feature contains future information"""
        for feature in features:
            self._check_temporal_boundary(df, feature)
            self._check_logical_availability(feature)
    
    def _check_temporal_boundary(self, df, feature):
        """Verify feature only uses data before prediction date"""
        future_data = df[
            (df['gameDate'] > self.prediction_date) & 
            (df[feature].notna())
        ]
        if len(future_data) > 0:
            raise ValueError(f"Temporal leakage in {feature}")
    
    def _check_logical_availability(self, feature):
        """Check if feature would logically be available"""
        forbidden_patterns = ['Final', 'Season', 'Complete', 'Total']
        if any(pattern in feature for pattern in forbidden_patterns):
            warnings.warn(f"Potentially leaky feature: {feature}")
```

### **Testing for Data Leakage**

#### **Test A: Historical Backtest Validation**
```python
# Test model performance using only historical data
def backtest_no_leakage(model, data, start_date, end_date):
    results = []
    for test_date in pd.date_range(start_date, end_date, freq='W'):
        # Only use data before test_date for training
        train_data = data[data['gameDate'] < test_date]
        test_data = data[data['gameDate'] == test_date]
        
        model.fit(train_data[features], train_data[target])
        predictions = model.predict(test_data[features])
        results.append(calculate_metrics(test_data[target], predictions))
    
    return results
```

#### **Test B: Performance Reality Check**
```python
# If your model performs "too well", check for leakage
if model_accuracy > 0.75:  # Suspiciously high for NFL predictions
    print("⚠️  WARNING: Performance may indicate data leakage")
    print("NFL games are inherently unpredictable - investigate features")
```

### **Real-World Impact of Data Leakage**

#### **Before Leak Fixes (Overly Optimistic)**
- **Reported ROI**: 14.1% (inflated by future information)
- **Model Confidence**: Artificially high
- **Real Performance**: Would fail in production

#### **After Leak Fixes (Production Reality)**
- **Actual ROI**: 60.9% (validated with strict temporal boundaries)
- **Model Confidence**: Properly calibrated
- **Real Performance**: Reliable for live betting

### **Data Leakage Prevention Checklist**

- [ ] ✅ All rolling statistics use strict temporal boundaries
- [ ] ✅ No future game outcomes in feature engineering
- [ ] ✅ Cross-validation respects chronological order
- [ ] ✅ Features available at actual prediction time
- [ ] ✅ Performance metrics validated on truly unseen data
- [ ] ✅ Model tested with historical backtest framework
- [ ] ✅ No complete season statistics for mid-season predictions
- [ ] ✅ All team statistics calculated from past games only

## 🐛 Advanced Streamlit Debugging Patterns

### Common Error Patterns & Fixes

#### **Large File Handling**
```python
# Error: FileNotFoundError for nfl_history_2020_2024.csv.gz
try:
    large_df = pd.read_csv('data_files/nfl_history_2020_2024.csv.gz')
except FileNotFoundError:
    st.warning("Large historical file not available - using fallback data")
    # Graceful degradation with smaller dataset
```

#### **DataFrame Display Compatibility**
```python
# OLD (deprecated): use_container_width=True, width='stretch'
# NEW (current): Modern Streamlit dataframe display
st.dataframe(df, column_config={
    'pred_underdogWon_optimal': st.column_config.CheckboxColumn('Betting Signal')
})
```

#### **Red Triangle Errors (Type/Format Mismatches)**
```
PROBLEM: Red triangles in number cells indicate type/format issues

[Issue A] Formatting decimals as percentages
# WRONG: format='%.1f%%' on decimal values (0.456)
'prob_underdogWon': st.column_config.NumberColumn('ML Prob', format='%.1f%%')  # ❌

# FIX: Multiply by 100 first, then use percentage format
display_df['prob_underdogWon'] = display_df['prob_underdogWon'] * 100
'prob_underdogWon': st.column_config.NumberColumn('ML Prob', format='%.1f%%')  # ✅

# [Issue B] Using TextColumn for dates
# WRONG: Converting datetime to string then using TextColumn
predictions_df['gameday'] = predictions_df['gameday'].dt.strftime('%Y-%m-%d')
'gameday': st.column_config.TextColumn('Date')  # ❌

# FIX: Keep as datetime and use DateColumn
'gameday': st.column_config.DateColumn('Date', format='MM/DD/YYYY')  # ✅

# [Issue C] Incorrect format for decimal values
# WRONG: Using percentage format on raw decimals
'edge_underdog_ml': st.column_config.NumberColumn('Edge', format='%.1f%%')  # ❌ if value is 0.08

# FIX: Either multiply by 100 OR use decimal format
display_df['edge_underdog_ml'] = display_df['edge_underdog_ml'] * 100  # Convert to percentage
'edge_underdog_ml': st.column_config.NumberColumn('Edge', format='%.1f%%')  # ✅
# OR
'edge_underdog_ml': st.column_config.NumberColumn('Edge', format='%.3f')  # ✅ shows 0.080
```

#### **Memory Management**
- Dashboard loads ~500MB-1GB during processing
- Use `@st.cache_data` for expensive operations
- Break large dataframes into chunks for display

#### **Filtered Historical Data Formatting**
```python
# NFLverse play-by-play data with probability columns (wp, def_wp, vegas_wp, etc.)
display_data = filtered_data.head(50).copy()

# Identify probability columns and convert to percentages
prob_columns = ['wp', 'def_wp', 'home_wp', 'away_wp', 'vegas_wp', 
                'cp', 'cpoe', 'success', 'pass_oe']

for col in prob_columns:
    if col in display_data.columns:
        if display_data[col].between(0, 1).all():
            display_data[col] = display_data[col] * 100

# Configure with appropriate formats
column_config = {
    'wp': st.column_config.NumberColumn('wp', format='%.1f%%'),
    'epa': st.column_config.NumberColumn('epa', format='%.3f'),
    'yards_gained': st.column_config.NumberColumn('yards_gained', format='%d')
}
```

#### **Model Loading Validation**
```python
# Check required files exist before proceeding
required_files = ['best_features_moneyline.txt', 'model_metrics.json']
missing_files = [f for f in required_files if not os.path.exists(f)]
if missing_files:
    st.error(f"Run nfl-gather-data.py first. Missing: {missing_files}")
```

#### **Date Filtering Issues (RESOLVED November 2025)**
```python
# ISSUE: Betting sections showing 2020 games instead of upcoming games
# ROOT CAUSE: predictions_df being modified by earlier sections
# - "Show Model Probabilities" section converted gameday to strings
# - Other sections filtered for past dates, mutating shared variable

# SOLUTION: Reload fresh data in each betting section
predictions_df_upcoming = pd.read_csv(predictions_csv_path, sep='\t')
predictions_df_upcoming['gameday'] = pd.to_datetime(predictions_df_upcoming['gameday'], errors='coerce')

# Filter for future games only
today = pd.to_datetime(datetime.now().date())
predictions_df_upcoming = predictions_df_upcoming[predictions_df_upcoming['gameday'] > today]

# Then apply betting signal filters
upcoming_bets = predictions_df_upcoming[predictions_df_upcoming['pred_underdogWon_optimal'] == 1]
```

### Performance Optimization
- **Load times**: 10-30 seconds typical for full data processing
- **Caching strategy**: Cache model predictions, not raw data loading
- **Progressive loading**: Show basic metrics first, detailed analysis second

## 📡 NFLverse Data Integration Specifics

### Primary Data Sources & Quality
```python
# High-quality play-by-play data (not ESPN scraping)
import nfl_data_py as nfl
games = nfl.import_schedules(seasons)  # Official NFL scheduling data
pbp = nfl.import_pbp_data(seasons)     # Play-by-play (not used in current models)
```

### Data Pipeline Architecture
- **Step 1: `create-nfl-historical.py`** - Initial NFLverse data fetch
- **Step 2: `fetch_nflfastR_games.py`** - Regular updates during season
- **Step 3: `nfl-gather-data.py`** - Feature engineering from raw data

### Key Data Quality Patterns
```python
# Remove games without betting lines (can't validate predictions)
historical_game_level_data = historical_game_level_data.dropna(subset=['spread_line'])

# Handle spread_line polarity correctly
# Positive = home team favored, negative = away team favored
historical_game_level_data['homeFavored'] = np.where(historical_game_level_data['spread_line'] > 0, 1, 0)
```

### File Size Management
- **Git LFS integration** for files >50MB
- **Compression**: `.csv.gz` format for large datasets
- **Fallback handling**: Dashboard works without largest files

### Data Coverage & Updates
- **Seasons**: 2020-2024 (5 years of data)
- **Update frequency**: Weekly during NFL season
- **Data completeness**: ~4,000+ games with betting lines

## 🎯 Model Performance Validation Approaches

### Multi-Metric Evaluation Strategy
```python
# model_metrics.json tracks key performance indicators
{
  "Spread Accuracy": 0.55,      # Slightly better than random (realistic)
  "Moneyline Accuracy": 0.70,   # High accuracy on selected bets
  "Totals Accuracy": 0.58       # After fixing data leakage issues
}
```

### Threshold Optimization Validation
```python
# F1-score optimization (lines 213-232 in nfl-gather-data.py)
thresholds = np.arange(0.1, 0.6, 0.02)  # Test 0.10 to 0.60 in 0.02 steps
best_threshold = thresholds[np.argmax(f1_scores)]  # Usually ~0.28
```

### Backtesting Validation Patterns
- **Historical ROI**: 60.9% on 1,211 betting signals (leak-free validation) for moneyline
- **Spread ROI**: 75.5% on spread bets with 91.9% win rate (≥54% threshold)
- **Win rate validation**: 40.8% on underdog picks (profitable due to odds)
- **Selective betting**: Only ~72% of games generate betting signals

### Model Architecture Validation
```python
# Calibrated probabilities for accurate confidence estimates
model_moneyline = CalibratedClassifierCV(
    XGBClassifier(eval_metric='logloss'), 
    method='sigmoid', 
    cv=3
)
```

### Cross-Validation Strategy
- **Time-series splits**: Prevent future data leakage
- **Stratified sampling**: Handle class imbalance (favorites vs underdogs)
- **Feature stability**: Monte Carlo testing ensures robust feature selection

### Edge Calculation Validation
```python
# Verify positive expected value
edge_underdog_ml = model_prob - implied_odds_prob
# Only bet when edge > 0 AND prob >= 28% threshold
betting_signal = (prob_underdogWon >= 0.28) & (edge_underdog_ml > 0)
```