# New Features Added - December 13, 2025

## Overview
Added momentum, rest advantage, and weather impact features to improve 2025 predictions. These features capture short-term team performance trends that should help the model make better predictions for upcoming games.

## Feature Categories

### 1. Momentum Features (Last 3 Games)
Captures recent performance trends to identify hot/cold teams:

- **homeTeamLast3WinPct**: Win percentage over last 3 games for home team
- **awayTeamLast3WinPct**: Win percentage over last 3 games for away team
- **homeTeamLast3AvgScore**: Average points scored in last 3 games (home team)
- **awayTeamLast3AvgScore**: Average points scored in last 3 games (away team)
- **homeTeamLast3AvgScoreAllowed**: Average points allowed in last 3 games (home team)
- **awayTeamLast3AvgScoreAllowed**: Average points allowed in last 3 games (away team)
- **homeTeamPointDiffTrend**: Point differential trend - positive = improving, negative = declining (home team)
- **awayTeamPointDiffTrend**: Point differential trend - positive = improving, negative = declining (away team)

**Impact**: Teams on winning streaks or with improving point differentials should have higher predicted win probabilities.

### 2. Rest Advantage Features
Captures the impact of days since last game (fatigue/preparation):

- **restDaysDiff**: Difference in rest days (home_rest - away_rest)
- **homeTeamWellRested**: 1 if home team has ≥10 days rest (0 otherwise)
- **awayTeamWellRested**: 1 if away team has ≥10 days rest (0 otherwise)
- **homeTeamShortRest**: 1 if home team has ≤6 days rest (0 otherwise)
- **awayTeamShortRest**: 1 if away team has ≤6 days rest (0 otherwise)

**Impact**: Teams with more rest typically have an advantage. Thursday night games (short rest) often favor defensive teams.

**Note**: Rest day data (`away_rest`, `home_rest`) was already in the dataset but not being fully utilized.

### 3. Weather Impact Features
Captures extreme weather conditions that affect game dynamics:

- **isColdWeather**: 1 if temperature ≤32°F (freezing)
- **isWindy**: 1 if wind speed ≥15 mph
- **isExtremeWeather**: 1 if temp ≤25°F OR wind ≥20 mph

**Impact**: 
- Cold weather reduces passing efficiency (favors running teams)
- Windy conditions affect field goal accuracy and passing games
- Extreme weather often leads to lower-scoring games (impacts over/under)

**Note**: Weather data (`temp`, `wind`) was already in the dataset but not being fully utilized.

## Implementation Details

### Data Leakage Prevention
All momentum features use the `calc_momentum_stat()` and `calc_point_diff_trend()` functions which:
- Only consider games BEFORE the current game (no lookahead)
- Filter by season and week to ensure chronological ordering
- Return 0 for teams with insufficient history

### Feature Engineering Functions

```python
def calc_momentum_stat(df, team_col, stat_col, num_games=3):
    """Calculate stat over last N games for momentum tracking"""
    # Returns average of stat_col over last N games for each team
    # Uses only prior games (no data leakage)

def calc_point_diff_trend(df, team_col, num_games=3):
    """Calculate if point differential is improving or declining"""
    # Positive = improving, negative = declining
    # Uses diff().mean() on last N games' point differentials
```

## Expected Impact

### Short-term (Immediate)
- **Better 2025 predictions**: Momentum features will help identify teams playing above/below their season averages
- **Rest advantage detection**: Should improve Thursday night game predictions (known rest disadvantage)
- **Weather-adjusted totals**: Cold/windy games should show lower over/under predictions

### Medium-term (After Retraining)
- **Higher confidence scores**: More features means better separation between good/bad bets
- **More +EV opportunities**: With better predictions, more games should cross the 52.4% breakeven threshold
- **Improved calibration**: More informative features should reduce prediction error

## Next Steps

1. **Retrain Models**: Run `python nfl-gather-data.py` to regenerate models with new features
2. **Update Feature Lists**: New features will be automatically evaluated and added to `best_features_*.txt` if important
3. **Monitor Performance**: Track if 2025 predictions improve (more games >52.4% confidence)

## Technical Notes

- Total new features: 18 (8 momentum + 5 rest + 3 weather + 2 existing weather variables fully utilized)
- Computational cost: Moderate (momentum features require iteration, ~30 seconds extra processing)
- Memory impact: Minimal (18 new columns in existing dataframe)
- No breaking changes: Existing features remain unchanged

## Success Metrics

After retraining, we should see:
- ✅ More 2025 games with >52.4% confidence (currently 0)
- ✅ Improved feature importance for momentum/rest/weather features
- ✅ Better calibration (predicted vs actual closer match)
- ✅ Higher ROI on historical backtesting

## File Changes

- **Modified**: `nfl-gather-data.py`
  - Added `calc_momentum_stat()` function
  - Added `calc_point_diff_trend()` function
  - Added 18 new feature calculations
  - Updated features list with new features
