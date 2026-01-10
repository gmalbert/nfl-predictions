# Model Performance Testing Scripts

These scripts help test and validate the Model Performance page with 2025 season data.

## Available Scripts

### 1. `check_weekly_stats_availability.py`
**Purpose**: Check if pre-aggregated weekly player stats are available from nflverse
**When to use**: Before running analysis to see if fast method is available
**Run**: `python scripts/check_weekly_stats_availability.py`

**Output**:
- ✅ If pre-aggregated stats are available (fast method)
- ❌ If only PBP aggregation is available (slower but works)

---

### 2. `test_backtest_single_week.py`
**Purpose**: Test backtest functionality for a single week (Week 18, 2025)
**When to use**: Quick validation that backtest module is working
**Run**: `python scripts/test_backtest_single_week.py`

**Output**:
- Total predictions evaluated
- Overall accuracy percentage
- ROI percentage
- Accuracy by prop type
- Accuracy by confidence tier

**Sample Output**:
```
✅ Week 18 Analysis Complete
   Predictions evaluated: 174
   Overall accuracy: 75.3%
   
   Accuracy by prop type:
      passing_tds: 80.0%
      receptions: 87.8%
      receiving_tds: 81.6%
```

---

### 3. `test_model_performance_all_weeks.py`
**Purpose**: Test backtest functionality across multiple weeks (1, 10, 18) of 2025 season
**When to use**: Comprehensive validation of Model Performance page
**Run**: `python scripts/test_model_performance_all_weeks.py`

**Output**:
- Accuracy and ROI for each week tested
- Top performing prop types per week
- Summary of data availability

**Sample Output**:
```
Week 1: 65.1% accuracy (24.2% ROI)
Week 10: 69.9% accuracy (33.4% ROI)
Week 18: 75.3% accuracy (43.7% ROI)
```

---

## How It Works

The backtest module (`player_props/backtest.py`) uses a **try/fallback approach**:

1. **First**: Try to load pre-aggregated weekly stats (fast, instant)
2. **Fallback**: If 404 error, aggregate play-by-play data (slower, ~5 seconds)

This ensures data availability while optimizing for speed when possible.

## Recent Changes (Jan 10, 2026)

✅ **Fixed**: 2025 season data now accessible via PBP aggregation
✅ **Smart Fallback**: Auto-switches between fast/slow methods
✅ **Future-Proof**: Will use pre-aggregated stats when available

## Data Sources

- **Pre-aggregated**: `nfl_data_py.import_weekly_data()` (preferred)
- **PBP Fallback**: `https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2025.parquet`

---

**Last Updated**: January 10, 2026
**Author**: GitHub Copilot
