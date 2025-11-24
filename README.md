# 🏈 NFL Betting Analytics & Predictions Dashboard

[📍 View the Product Roadmap](./ROADMAP.md)

A sophisticated NFL analytics platform that uses advanced machine learning to identify profitable betting opportunities. This system analyzes historical NFL data with **50+ enhanced features** across multiple time scales to predict game outcomes and provides actionable betting insights with proven **60.9% ROI** performance on data-leakage-free models.

**Now featuring a multi-page Streamlit app** with dedicated Historical Data page for advanced filtering and analysis of 196k+ play-by-play records.

## 📋 Table of Contents

- [🎯 Key Features](#-key-features)
  - [📊 Interactive Dashboard](#-interactive-dashboard)
  - [💰 Proven Betting Strategy](#-proven-betting-strategy)
  - [🤖 Advanced Machine Learning](#-advanced-machine-learning)
- [📈 Model Performance](#-model-performance-data-leakage-free)
- [📊 Data Sources](#-data-sources)
  - [Primary Data: NFLverse](#primary-data-nflverse)
  - [Betting Lines: ESPN](#betting-lines-espn)
- [🎮 How to Use](#-how-to-use)
  - [Running the System](#1-running-the-system)
  - [Dashboard Sections](#2-dashboard-sections)
- [📁 Enhanced Features](#-enhanced-features)
  - [Current Season Performance Tracking](#current-season-performance-tracking)
  - [Historical Season Context](#historical-season-context)
  - [Head-to-Head Matchup History](#head-to-head-matchup-history)
- [🔧 Technical Architecture](#-technical-architecture)
  - [Machine Learning Pipeline](#machine-learning-pipeline)
  - [Data Engineering](#data-engineering)
  - [Betting Strategy Architecture](#betting-strategy-architecture)
- [📁 Recent Updates](#-recent-updates-october-2025)
- [🎯 Getting Started](#-getting-started)
- [🔧 Troubleshooting](#-troubleshooting)
- [⚠️ Responsible Gambling Notice](#️-responsible-gambling-notice)
- [🤝 Contributing](#-contributing)

## 🎯 Key Features

### 📊 **Interactive Dashboard**
- **🔥 Next 10 Underdog Bets**: Actionable moneyline betting opportunities with complete payout calculations
- **🎯 Next 10 Spread Bets**: NEW high-performance spread betting section with 91.9% historical win rate
- **Confidence Tiers**: Visual indicators (🔥 Elite, ⭐ Strong, 📈 Good) for bet prioritization
- **Live Game Predictions**: Real-time probability calculations for upcoming NFL games
- **Enhanced Betting Signals**: Dual-strategy recommendations with proper spread/moneyline logic
- **Performance Tracking**: Historical betting performance with ROI calculations and survivorship bias awareness
- **Edge Analysis**: Compare model predictions against sportsbook odds to find value bets
- **Monte Carlo Feature Selection**: Interactive experimentation with feature combinations for model optimization
- **Enhanced Reliability**: Robust error handling and graceful fallbacks for uninterrupted analysis

### 💰 **Proven Betting Strategy**
- **Triple Strategy Success**: Moneyline (65.4% ROI) + Spread (75.5% ROI) + Over/Under betting
- **Elite Spread Performance**: 91.9% win rate on high-confidence spread bets (≥54% threshold)
- **Moneyline Strategy**: 59.5% win rate on underdog picks with 28% F1-optimized threshold
- **Over/Under Model**: NEW totals betting with F1-optimized thresholds and value edge calculations
- **Professional-Grade Validation**: Data leakage eliminated, realistic performance metrics
- **Selective Betting**: High-confidence filtering for maximum profitability

### 🤖 **Advanced Machine Learning**
- **Three Specialized Models**: Separate XGBoost models for spread, moneyline, and over/under predictions
- **F1-Score Optimization**: All models use F1-score maximization to find optimal betting thresholds
- **Optimized XGBoost Models** with production-ready hyperparameters and probability calibration
- **Enhanced Monte Carlo Feature Selection** testing 200 iterations with 15-feature subsets
- **Data Leakage Prevention**: Strict temporal boundaries ensuring only pre-game information
- **Class Balancing** with computed scale weights for imbalanced datasets
- **Multi-Target Prediction**: Spread (56.3%), moneyline (64.2%), totals (56.2%) accuracy

[⬆️ Back to Top](#-nfl-betting-analytics--predictions-dashboard)

## 📈 **Model Performance** (Data Leakage Free)

| Prediction Type | Cross-Val Accuracy | Betting Performance | ROI | Key Insight |
|----------------|-------------------|---------------------|-----|-------------|
| **Spread** | 58.9% | **91.9% win rate** (≥54% threshold) | **75.5%** | Elite performance through selective betting |
| **Moneyline** | 64.2% | 59.5% win rate (≥28% threshold) | **65.4%** | Strong underdog value identification |
| **Over/Under** | 56.2% | F1-optimized thresholds | TBD | NEW - Totals betting with value edge analysis |

### **Major Performance Breakthrough (November 2025)**
- **Spread Model Fixed**: Corrected inverted predictions from 3.6% to 91.9% win rate
- **Dual Strategy Success**: Both spread and moneyline betting now highly profitable
- **Data Leakage Free**: Strict temporal boundaries ensure production reliability
- **Survivorship Bias Awareness**: 91.9% rate is on selective 33% of games, not overall performance
- **Confidence Calibration**: Model knows when it's likely to be right vs wrong

## 📊 **Data Sources**

### **Primary Data: NFLverse**
- **Source**: [NFLverse Project](https://github.com/nflverse/nflverse-data) - The most comprehensive NFL dataset available
- **Coverage**: Play-by-play data from 2020-2024 seasons
- **Data Quality**: Professional-grade data used by NFL analysts and researchers
- **Update Frequency**: Updated weekly during NFL season

### **Betting Lines Data: ESPN**
- **Source**: ESPN NFL scores and betting data
- **Includes**: Point spreads, moneylines, over/under totals, and odds
- **Coverage**: Historical betting lines for model training and backtesting

### **Team Statistics & Features**
- **All-Time Rolling Statistics**: Win percentages, point differentials, scoring averages
- **Current Season Performance**: Weekly updated team form and scoring trends  
- **Historical Season Context**: Prior season records for baseline team quality
- **Head-to-Head Matchups**: Team-specific historical performance data
- **Situational Data**: Home/away performance, division games, weather conditions
- **Advanced Metrics**: Blowout rates, close game performance, coaching records

[⬆️ Back to Top](#-nfl-betting-analytics--predictions-dashboard)

## 🎮 **How to Use**

### **1. Running the System**
```bash
# Build historical data and train models (single-step):
python build_and_train_pipeline.py

# Or run the steps separately:
# Train the models (build features and train)
python nfl-gather-data.py

# Launch the dashboard
streamlit run predictions.py
```

### **2. Dashboard Sections**

The dashboard uses **modern multi-page navigation** with a main predictions page and dedicated historical data page. Each section displays data with **professional formatting** including percentages, proper date formats, and descriptive column labels.

#### **🏈 Main Page: Predictions**
The primary predictions interface with tab-based navigation for betting analysis.

#### **📊 Tab: Model Predictions**
- **Model Predictions vs Actual Results**: Historical game outcomes with checkbox indicators
  - Formatted columns: Game Date (MM/DD/YYYY), Team names, Scores, Spread/O/U lines
  - Checkbox columns show predicted vs actual spread coverage and totals
  - Displays 50 most recent completed games with proper date formatting

#### **🎯 Tab: Probabilities & Edges**
- **Upcoming Game Probabilities**: Shows betting opportunities with model confidence
  - **Percentage Display**: All probabilities shown as percentages (e.g., "45.6%" instead of "0.456")
  - **Spread Probabilities**: Model confidence underdog will cover spread
  - **Moneyline Probabilities**: Model confidence underdog wins outright
  - **Over/Under Probabilities**: Model confidence for totals betting
  - **Edge Calculations**: Value identification (model % - implied %)
  - **Compact Labels**: "Spread Prob", "ML Prob", "Over Edge" with helpful tooltips

#### **💰 Tab: Betting Performance**
- **Key Metrics Table**: Clean, organized display of model performance
  - Spread, Moneyline, and Totals accuracy (formatted to 3 decimals)
  - Mean Absolute Error for each model
  - Optimal thresholds (28% for ML, 54% for spread)
  - Compact 600px width for easy scanning
- **Performance Tracking**: Historical betting performance with ROI calculations
- **Survivorship Bias Awareness**: Transparent about selective betting strategy

#### **🔥 Tab: Underdog Bets**
- **Next 10 Underdog Betting Opportunities**: Moneyline strategy recommendations
- **Chronological order**: Upcoming games where model has ≥28% confidence
- **Complete betting info**: Favored team, underdog, spread, model confidence, expected payout
- **Real payout calculations**: Exact profit amounts for $100 bets using live moneyline odds
- **Example**: "Vikings (H) +180 ($180 profit on $100)" when Chargers are favored

#### **📈 Tab: Spread Bets** ⭐ *ELITE PERFORMANCE*
- **Next 15 Spread Betting Opportunities**: High-confidence spread recommendations
- **Confidence Tiers**: 🔥 Elite (75%+), ⭐ Strong (65-74%), 📈 Good (54-64%)
- **Historical Performance**: 91.9% win rate on high-confidence bets, 75.5% ROI
- **Smart Sorting**: Ordered by confidence level for optimal bet selection
- **Spread Explanation**: Team can lose game but still "cover" (e.g., lose by less than spread)

#### **🎯 Tab: Over/Under Bets** ⭐ *NEW*
- **Totals Betting Opportunities**: Model predictions for over/under betting
- **Confidence Tiers**: Elite (≥65%), Strong (60-65%), Good (55-60%), Standard (<55%)
- **Value Edge Calculation**: Expected profit percentage based on model probability vs odds
- **Smart Bet Selection**: Recommendations sorted by value edge for optimal selection
- **Complete Payout Info**: Expected returns on $100 bets for both over and under options

#### **📋 Tab: Betting Log**
- **Automated Bet Tracking**: Logs all betting recommendations with timestamps
- **Results Integration**: Automatically updates with game outcomes
- **Performance Analysis**: Win/loss tracking for accountability

#### **📊 Historical Data Page** ⭐ *NEW MULTI-PAGE APP*
- **Dedicated Page**: Separate navigation page for historical data analysis
- **196k+ Play Records**: Complete play-by-play data from 2020-2024 seasons
- **Advanced Filtering**: 12+ filter controls including:
  - Team filters (offense/defense)
  - Game context (down, quarter, play type)
  - Field position sliders (yards to go, yardline)
  - Score filters (differential, team scores)
  - Advanced metrics (EPA, win probability)
- **Reset Functionality**: One-click filter reset with session state management
- **Paginated Display**: 50-500 rows per page with navigation controls
- **Rich Column Configuration**: Formatted dates, percentages, checkboxes for play outcomes
- **Back to Predictions**: Easy navigation button to return to main page

#### **⚙️ Additional Features**
- **Feature Importances**: Top model features with mean/std importance (3 decimals)
- **Monte Carlo Results**: Feature selection testing with formatted metrics
- **Filtered Historical Data**: Play-by-play data with percentage formatting for win probabilities
- **Error Metrics**: Model accuracy and MAE displayed with consistent 3-decimal formatting

## 💡 **Betting Strategy Explained**

### **The Core Insight**
Traditional betting advice says "always bet favorites" (70% win rate), but this system finds **value in selective underdog betting**:

1. **Model identifies underdogs** with higher win probability than betting odds suggest
2. **24% threshold optimization** maximizes F1-score for better long-term profitability
3. **Risk management** through selective betting (only ~72% of games get betting signals)

### **Example Betting Signal**
```
Game: Chiefs @ Raiders
Model Probability (Underdog Win): 30%
Sportsbook Implied Probability: 22%
Betting Signal: 1 (BET ON RAIDERS)
Threshold: ≥28% (F1-optimized, leak-free)
Expected Value: Positive due to 8% edge
ROI Expectation: 60.9% based on historical performance
```

## 🔬 **Technical Features**

### **Model Architecture**
- **Optimized XGBoost Classifiers** with production-tuned hyperparameters:
  - `n_estimators=300`, `learning_rate=0.05`, `max_depth=6`
  - L1/L2 regularization (`reg_alpha=0.1`, `reg_lambda=1.0`)
  - Subsampling (`subsample=0.8`, `colsample_bytree=0.8`)
- **Calibrated Probability Outputs** using sigmoid/isotonic scaling
- **Enhanced Feature Engineering**: 50+ leak-free temporal features
- **Time-Series Aware Cross-Validation** preventing future data leakage

### **🆕 Enhanced Predictive Features (Latest Update)**

#### **Current Season Performance Tracking**
- **Real-time Team Form**: `homeTeamCurrentSeasonWinPct`, `awayTeamCurrentSeasonWinPct`
- **Scoring Trends**: `homeTeamCurrentSeasonAvgScore`, `awayTeamCurrentSeasonAvgScore`
- **Defensive Performance**: `homeTeamCurrentSeasonAvgScoreAllowed`, `awayTeamCurrentSeasonAvgScoreAllowed`

#### **Historical Season Context**
- **Prior Season Records**: `homeTeamPriorSeasonRecord`, `awayTeamPriorSeasonRecord`
- **Annual Stability Metrics**: Complete previous season win percentages for baseline team quality

#### **Head-to-Head Matchup History**
- **Team-Specific Advantages**: `headToHeadHomeTeamWinPct`
- **Historical Matchup Performance**: How teams have performed against each other over time

#### **Technical Implementation**
- **Temporal Integrity**: All features maintain strict data leakage prevention
- **Multi-Scale Analysis**: Current season trends + historical season records + specific matchup history
- **Automated Integration**: Features automatically included in Monte Carlo feature selection

### **Data Pipeline**
- **Automated Feature Creation**: Rolling averages, win percentages, trends with strict temporal controls
- **Data Leakage Prevention**: All statistics use only prior games (`season < current` OR `week < current`)
- **Enhanced Validation**: Real-time feature availability and XGBoost compatibility checks
- **Quality Assurance**: Removes games with missing betting lines, validates data integrity
- **Threshold Optimization**: F1-score maximization finds optimal decision boundary (28%)
- **Robust Error Handling**: Graceful fallbacks for missing data and feature inconsistencies
- **Production Ready**: No future information leakage, realistic performance expectations

[⬆️ Back to Top](#-nfl-betting-analytics--predictions-dashboard)

## 📁 **Project Structure**

```
nfl-predictions/
├── nfl-gather-data.py      # Main model training script
├── predictions.py          # Streamlit dashboard
├── data_files/            # Data storage directory
```

## 🔧 **Technical Architecture**

### **Machine Learning Pipeline**
- **Primary Algorithm**: XGBoost with production-optimized parameters
  - **Training Parameters**: 300 estimators, 0.05 learning rate, max depth 6
  - **Regularization**: L1=1, L2=1 for overfitting prevention  
  - **Monte Carlo Parameters**: 100 estimators, 0.1 learning rate for feature selection
- **Model Calibration**: CalibratedClassifierCV with sigmoid method for accurate probabilities
- **Feature Selection**: Monte Carlo optimization (200 iterations, 15-feature subsets)
- **Cross-Validation**: Time-series aware splits preventing data leakage
- **Class Handling**: Weighted models addressing favorite/underdog imbalance (~70/30 split)

### **Data Engineering**
- **Primary Source**: NFLverse (nfl_data_py) - official NFL play-by-play data
- **Betting Data**: ESPN odds and lines (2020-2024 seasons)  
- **Feature Types**: Rolling statistics, head-to-head records, seasonal performance
- **Data Integrity**: Strict temporal boundaries prevent future data leakage
- **Data Quality**: ~4,000+ games with complete betting line information
- **Storage**: Git LFS integration for large datasets (>50MB)

### **Betting Strategy Architecture**
- **Threshold Optimization**: F1-score maximization finds 28% probability threshold (not 50%)
- **Selective Betting**: ~72% of games generate betting signals (selective strategy)
- **Edge Calculation**: `model_prob - implied_odds_prob` for value identification
- **ROI Focus**: Optimizes for profit margin, not raw accuracy percentage

[⬆️ Back to Top](#-nfl-betting-analytics--predictions-dashboard)

## 📁 **Recent Updates (November 2025)**

### **🔔 In-App Notification System (Latest)**
- **Problem Solved**: Users miss high-value betting opportunities when they open the app.
- **Solution**: Automatic, actionable in-app notifications that highlight Elite and Strong opportunities:
  - **Elite notifications** (🔥): bets with ≥65% model confidence
  - **Strong notifications** (⭐): bets with 60–65% model confidence
  - Notifications are shown as toasts and are deduplicated using `st.session_state` so the same game doesn't re-notify in the same session.
  - Each alert links to a per-alert page (query param `?alert=<guid>`) that shows friendly bet details, logos, bet type, gameday/time, and a small recommendation table.
  - The app also persists a detected public base URL (when available) into `data_files/app_config.json` so external tools (and the RSS generator) can create working per-alert links.
- **Impact**: Improves engagement by surfacing high-value bets and providing one-click access to full alert details.
- **Technical**: Uses `st.toast()` for toasts, session-state deduplication, per-alert pages rendered from `data_files/betting_recommendations_log.csv`, and persisted base-URL for external link generation.

### **🔔 RSS Feed & Rebuild Button**
- **Purpose**: Provide external notifications via RSS with per-alert links pointing back to the app.
- **Implementation**:
  - `scripts/generate_rss.py` builds `data_files/alerts_feed.xml`. It prefers the persisted `app_base_url` in `data_files/app_config.json` (if present), otherwise falls back to the `ALERTS_SITE_URL` environment variable or `http://localhost:8501/`.
  - The running app exposes a sidebar `"🔁 Rebuild RSS"` button that runs the generator in-place and reports success/errors.
- **Files**:
  - Persisted config: `data_files/app_config.json` (contains `{"app_base_url": "https://..."}` when detected)
  - RSS output: `data_files/alerts_feed.xml`

### **🎨 UI Layout Optimization (Latest)**
- **Problem Solved**: Excessive vertical space at the top of the main page reduced available screen real estate
- **Solutions Implemented**:
  - **Compact Header Layout**: Logo and title now arranged in columns [1, 4] instead of stacked vertically
  - **Logo Size Reduction**: Logo reduced from 250px to 150px width for better proportions
  - **Logo Positioning**: Logo positioned at top-left of its column with minimal spacing
  - **Loading Progress Optimization**: Reduced verbose loading messages and progress indicators
  - **Debug Message Cleanup**: Removed unnecessary debug print statements from UI
- **Impact**: Significantly more content visible above the fold, improved user experience
- **Technical**: Uses `st.columns()` for responsive layout and optimized spacing

### **🧠 Memory Optimization & Streamlit Cloud Deployment (Latest)**
- **Problem Solved**: App exceeded Streamlit Cloud resource limits due to high memory usage (1.5GB+)
- **Solutions Implemented**:
  - **Data Type Optimization**: `float32` for numeric columns (50% memory reduction), `Int8` for boolean columns
  - **DataFrame Views**: Replaced `.copy()` operations with views to eliminate memory duplication
  - **Lazy Loading**: All data loading uses `@st.cache_data` decorators with lazy initialization
  - **Pagination**: Added pagination for large datasets (>10k rows) with user warnings
  - **Spinner Configuration**: Suppress cache messages on technical pages, show on main UI
  - **Variable Cleanup**: Added `del` statements to clean up progress bars and temporary variables
- **Impact**: Reduced memory usage while maintaining full functionality, enabled Streamlit Cloud deployment
- **Technical**: Memory-efficient patterns prevent resource limit violations

### **⚙️ Cache Management UI (Latest)**
- **Problem Solved**: Users couldn't manually refresh cached data or check cache status
- **Solution**: Added Settings panel in sidebar with:
  - "🔄 Refresh Data" button to clear all cached data and reload
  - Helpful tooltips explaining functionality
- **Impact**: Gives users control over data freshness
- **Technical**: Uses `st.cache_data.clear()` for cache management

### **💰 Bankroll Management Tool (Latest)**
- **Problem Solved**: Users didn't know how much to bet on each recommendation for optimal risk management
- **Solution**: Added dedicated "💰 Bankroll Management" tab with:
  - Bankroll input field with configurable amounts ($100-$1M+)
  - Risk tolerance selector (Conservative 1%, Moderate 2%, Aggressive 3%, Very Aggressive 5%)
  - Elite bet identification (≥65% confidence threshold)
  - Position sizing calculator showing recommended bet amounts
  - Expected payout calculations for each bet
  - Expected value analysis (positive EV bets only)
  - Bankroll impact tracking (total exposure percentage)
- **Impact**: Enables responsible betting with Kelly Criterion-inspired position sizing
- **Technical**: Filters predictions for elite opportunities and calculates optimal bet sizes

### **📈 Model Performance Dashboard (Latest)**
- **Problem Solved**: Users couldn't see if betting recommendations were actually accurate
- **Solution**: Added dedicated "📈 Model Performance" tab with:
  - Overall metrics: Total bets, win rate, ROI, units won
  - Performance breakdown by confidence tier (Elite/Strong/Good/Standard)
  - Weekly performance tracking with line charts
  - Best performing bet types analysis
- **Impact**: Builds trust and transparency in model performance
- **Technical**: Reads from `betting_recommendations_log.csv` and calculates ROI metrics

### **⚡ Loading Progress Indicators (Latest)**
- **Problem Solved**: Users experienced 5-10 second load times with no feedback on progress
- **Solution**: Added detailed progress bar showing specific loading steps:
  - 25%: "Loading historical games..." - loads game-level data with predictions
  - 50%: "Loading model predictions..." - loads betting predictions CSV
  - 75%: "Loading play-by-play data..." - loads historical play-by-play records
  - 100%: "Ready!" - shows completion with brief pause
- **Impact**: Significantly improved perceived performance and user experience
- **Technical**: Uses Streamlit progress bar with text updates and automatic cleanup

### **⚡ Performance Optimizations for Historical Data (Latest)**
- **Memory-Efficient Data Types**: Reduced memory usage by ~50% using float32 and Int8 instead of float64/int64
- **Eliminated Data Copying**: Removed unnecessary `.copy()` operations on 196k row dataset
- **DataFrame Views**: Filter operations now use views instead of copies for instant performance
- **Streamlit Cloud Config**: Added `.streamlit/config.toml` with increased message size limits (500MB) and optimizations
- **Smart Warnings**: Alert users when displaying large result sets to encourage filtering
- **Result**: Historical Data page now loads significantly faster on Streamlit Cloud, avoiding timeout issues

### **🎯 NEW: Over/Under Betting Model**
- **Three-Model System**: Added dedicated over/under (totals) betting predictions alongside spread and moneyline
- **F1-Score Optimization**: Optimal threshold calculation for over/under predictions using F1-score maximization
- **Value Edge Analysis**: Calculates expected profit percentage based on model probability vs betting odds
- **Confidence Tiers**: Elite/Strong/Good/Standard classification for bet prioritization
- **Complete Payout Calculations**: Shows expected returns for both over and under bets on each game
- **Integrated Dashboard**: New "🎯 Over/Under Bets" tab with top 15 opportunities sorted by value edge

### **📊 NEW: Multi-Page Streamlit App**
- **Dedicated Historical Data Page**: Separate navigation page for 196k+ play-by-play records with advanced filtering
- **Clean Interface**: Main predictions page focuses on betting analysis, historical data on separate page
- **Easy Navigation**: "🏈 Back to Predictions" button for quick return to main page
- **12+ Filter Controls**: Team filters, game context, field position, scores, and advanced metrics
- **Session State Reset**: Reliable filter reset functionality using flag pattern
- **Pagination**: Display 50-500 rows per page with navigation controls

### **🔧 Bug Fixes & System Improvements (Latest)**
- **Fixed Over/Under Column Names**: Corrected KeyError for `pred_totalsProb` → `prob_overHit`
- **Added Moneyline Return Calculation**: Implemented missing `moneyline_bet_return` column computation
- **Fixed Indentation Issues**: Resolved Python syntax errors in complex nested tab structures
- **Improved Error Handling**: Better validation for column existence before accessing dataframe columns
- **Type Safety**: Enhanced Pylance compatibility with proper variable extraction patterns

### **🔥 CRITICAL: Data Leakage Elimination (October 2025)**
- **Issue Discovered**: Historical statistics were using ALL-TIME data (including future games during training)
- **Impact**: Models appeared to have 70%+ accuracy but would fail in production
- **Solution**: Implemented strict temporal boundaries - only prior games used for all statistics
- **Result**: More realistic 56-64% accuracy BUT **60.9% ROI** (up from 27.8%) due to higher quality signals
- **Production Ready**: Models now perform consistently with live data

### **⚡ Optimal XGBoost Parameters Implementation (October 2025)**
- **Upgraded**: From basic `eval_metric='logloss'` to production-tuned hyperparameters
- **Parameters**: 300 estimators, 0.05 learning rate, depth 6, with L1/L2 regularization
- **Benefits**: Better generalization, reduced overfitting, more stable predictions
- **Monte Carlo**: Separate lighter parameters for faster feature selection (100 estimators, depth 4)

### **🎯 Enhanced Monte Carlo Feature Selection (October 2025)**
- **Upgraded**: From 8-feature subsets to 15-feature subsets for better coverage
- **Iterations**: Increased from 100 to 200 iterations for more thorough optimization
- **Results**: Found optimal 7-8 feature models through comprehensive search space

### **🔥 Streamlit Dashboard Enhancements (October 2025)**
- **Next 10 Underdog Bets**: New prominent section showing actionable betting opportunities
  - Chronological order of next 10 recommended underdog bets
  - Complete betting info: favored team, underdog, spread, model confidence
  - Real payout calculations with exact profit amounts for $100 bets
- **Enhanced Recent Bets Display**: Added "Favored" column showing who sportsbooks favored
- **Corrected Spread Logic**: Fixed favorite/underdog identification (spread_line interpretation)
- **Improved User Experience**: Clear explanations, better formatting, actionable insights
- **Quality**: Better feature combinations leading to improved model stability

### **✅ Fixed Threshold Documentation (October 2025)**
- **Discovered**: Dashboard was showing inconsistent threshold information
- **Fixed**: Updated to reflect actual F1-optimized threshold (now 28% after leakage fixes)
- **Impact**: Users now see accurate betting strategy information

### **✅ Streamlit Compatibility Updates (October 2025)**
- **Fixed**: Deprecated `use_container_width` and `width='stretch'` parameters
- **Updated**: All dataframe displays now use modern Streamlit best practices
- **Result**: Dashboard works with latest Streamlit versions without errors

### **✅ Date Filtering Bug Fix (October 2025)**
- **Issue**: "Next 10 Underdog Bets" and "Next 10 Spread Bets" sections showing 2020 games instead of upcoming games
- **Root Cause**: `predictions_df` variable was being modified by earlier dashboard sections (converting gameday to strings, filtering for past dates)
- **Solution**: Each betting section now reloads fresh data from CSV and properly filters for future games only (`gameday > today`)
- **Impact**: Betting recommendations now correctly display only upcoming games in chronological order
- **Technical**: Fixed variable mutation issues across Streamlit sections through data isolation

### **✅ Git LFS Integration (October 2025)**
- **Added**: Large file support for `nfl_play_by_play_historical.csv.gz` (73.95MB)
- **Benefit**: Enables deployment to Streamlit Cloud with access to large datasets
- **Setup**: Properly configured `.gitignore` and `.gitattributes` for optimal repo management

### **🆕 Enhanced Feature Engineering (October 2025)**
- **Current Season Performance**: Added real-time team form tracking within current season
- **Prior Season Records**: Incorporated previous season's final win percentages for baseline metrics
- **Head-to-Head History**: Added historical matchup performance between specific teams
- **Technical Benefits**: Multi-scale temporal analysis with strict data leakage prevention
- **Model Impact**: Richer context for predictions across multiple time horizons

### **🔧 System Reliability & Error Resolution (October 2025)**
- **Fixed All KeyError Issues**: Resolved feature list mismatches between training and dashboard
- **Synchronized Feature Sets**: Ensured consistency across all 50+ features in both `nfl-gather-data.py` and `predictions.py`
- **Enhanced Monte Carlo Selection**: Fixed feature sampling to only use available numeric features
- **Robust Model Retraining**: Updated dashboard to handle dynamic feature selection correctly
- **Data Pipeline Validation**: Added comprehensive checks for feature availability and data integrity
- **Graceful Error Handling**: System now handles missing features and data inconsistencies smoothly

### **⚡ Performance & Stability Improvements**
- **Optimized Feature Loading**: Streamlined data processing for faster dashboard startup times
- **Memory Management**: Improved handling of large datasets with better resource utilization
- **Cross-Platform Compatibility**: Enhanced PowerShell and terminal command support
- **Real-Time Validation**: Added live feature availability checking during Monte Carlo experiments
- **Automated Fallbacks**: System degrades gracefully when optional features are unavailable

[⬆️ Back to Top](#-nfl-betting-analytics--predictions-dashboard)

## 🎯 **Getting Started**

1. **Install Dependencies**
   ```bash
   pip install streamlit pandas numpy xgboost scikit-learn
   ```

2. **Train Initial Models**
  ```bash
  # Single-step: build historical data and train models
  python build_and_train_pipeline.py
  ```

3. **Launch Dashboard**
   ```bash
   streamlit run predictions.py
   ```

4. **Deploy to Streamlit Cloud** (Recommended)
   - All data files are committed to repository
   - Memory optimizations ensure compatibility with Streamlit Cloud resource limits
   - Python 3.12 required for deployment

5. **Start Analyzing**
   - Check "Show Betting Analysis & Performance" for ROI metrics
   - Look for `pred_underdogWon_optimal = 1` in upcoming games
   - Use the edge calculations to find the best value bets

## 🔧 **Troubleshooting**

### **Common Issues**

**"TypeError: 'str' object cannot be interpreted as an integer"**
- **Cause**: Old Streamlit version compatibility issue
- **Fix**: Updated in recent version - use `git pull` to get latest fixes

**"Missing file: nfl_play_by_play_historical.csv.gz"**
- **Cause**: Large file not available locally
- **Fix**: Dashboard includes error handling and fallback data display

**"Betting signals don't match 30% threshold"**
- **Cause**: Documentation was incorrect (now fixed)
- **Reality**: Model uses 24% F1-optimized threshold, not 30%

### **Performance Notes**
- **Training time**: ~5-10 minutes with optimized parameters (300 estimators)
- **Monte Carlo time**: ~3-5 minutes for feature selection (200 iterations, 15-feature subsets)  
- **Dashboard load time**: ~10-30 seconds for full data processing
- **Memory usage**: ~1.5GB during data loading (optimized with float32/Int8 dtypes and DataFrame views)
- **Streamlit Cloud**: Fully compatible with memory optimizations for cloud deployment

[⬆️ Back to Top](#-nfl-betting-analytics--predictions-dashboard)

## 🔧 **Troubleshooting**

### **Common Issues & Solutions**

-#### **KeyError: Features not in index**
- **Cause**: Feature list mismatch between training and dashboard files
- **Solution**: Run `python build_and_train_pipeline.py` (or `python nfl-gather-data.py` to run only the training step) to regenerate feature files and ensure synchronization
- **Prevention**: Both files now auto-sync feature lists to prevent future mismatches

#### **"TypeError: 'str' object cannot be interpreted as an integer"**
- **Cause**: Old Streamlit version compatibility issue
- **Solution**: System now includes modern Streamlit compatibility - use `git pull` for latest fixes

#### **"Missing file: nfl_play_by_play_historical.csv.gz"**
- **Cause**: Large file not available locally (Git LFS)
- **Solution**: Dashboard includes graceful error handling and fallback data display
- **Alternative**: Download manually or use Git LFS: `git lfs pull`

#### **Monte Carlo Feature Selection Errors**
- **Cause**: Trying to sample features not available in processed dataset
- **Solution**: Fixed - system now validates feature availability before sampling
- **Features**: Enhanced error handling with automatic feature filtering

#### **ValueError: DataFrame.dtypes for data must be int, float, bool or category**
- **Cause**: XGBoost receiving non-numeric columns (object data types like team names, coaches)
- **Solution**: Fixed - automatic filtering to numeric/boolean/categorical data types only
- **Prevention**: All model inputs now validated for XGBoost compatibility before training

#### **Dashboard Won't Load / Port Already in Use**
- **Solution**: Use different port: `streamlit run predictions.py --server.port=XXXX`
- **Default ports**: Try 8501, 8502, 8503, etc.
- **Terminal**: Check for running Streamlit processes with `tasklist | findstr streamlit`

#### **Betting Sections Show Old Games (2020) Instead of Upcoming**
- **Cause**: Variable mutation across Streamlit sections - `predictions_df` modified by earlier filters
- **Solution**: Fixed in latest version - sections now reload fresh data and filter properly
- **Verification**: Betting recommendations should show games with future dates only
- **Technical**: Each section uses isolated dataframe copy to prevent cross-section contamination

### **Performance Optimization Tips**
- **Faster Loading**: Use `@st.cache_data` decorator for expensive operations (already implemented)
- **Memory Management**: Dashboard automatically handles large datasets with chunking
- **Feature Selection**: Start with smaller Monte Carlo iterations (50-100) for faster testing

## ⚠️ **Responsible Gambling Notice**

This tool is for educational and analytical purposes. While our backtesting shows strong historical performance:
- **Past performance doesn't guarantee future results**
- **Only bet what you can afford to lose**
- **Consider this one factor in your betting decisions**
- **Gambling involves risk - bet responsibly**

## 🤝 **Contributing**

This project welcomes contributions! Areas for improvement:
- Additional data sources (weather, injuries, etc.)
- Enhanced feature engineering
- Alternative modeling approaches
- UI/UX improvements

---

[⬆️ Back to Top](#-nfl-betting-analytics--predictions-dashboard)

**Built with**: Python • Streamlit • XGBoost • Scikit-learn • NFLverse Data

## 📥 Export Downloads & Sidebar (Nov 23, 2025)

- **Where to find exports**: The app exposes always-visible download controls in the sidebar for:
  - `Predictions CSV` — current predictions shown in the app
  - `Betting Log` — the `betting_recommendations_log.csv` file
- **Why you might not see them**: The Streamlit sidebar can be collapsed by default. If you do not see the download buttons, click the small chevron in the top-left of the page to expand the sidebar.
- **Behavior**: Download buttons are rendered after the app finishes loading data (the app reserves lightweight placeholders while data loads and populates the real `st.download_button` once `predictions_df` and the betting log are available).

## 🧪 Smoke Test & Quick Run

To validate the core imports and data-loading behavior without running the full Streamlit UI, use the included smoke test:

```bash
# Run a lightweight import-and-load smoke test
python smoke_test.py
```

This prints basic progress and a final `SMOKE OK` line when successful. For full interactive use:

```bash
python build_and_train_pipeline.py   # builds predictions/data if needed (single-step)
streamlit run predictions.py
```