# Player Props Feature Roadmap

## Overview
Enhancement roadmap for NFL Player Props prediction system. Prioritized by impact and implementation complexity.

---

## 🔥 High Priority Improvements

### 1. Opponent Defense Rankings
**Impact**: High | **Complexity**: Medium | **Timeline**: 1-2 days

Add defensive rankings to improve prediction accuracy by accounting for matchup difficulty.

```python
# In player_props/predict.py - Add to get_player_features()

def get_opponent_defense_rank(opponent, stat_type, all_stats):
    """
    Calculate opponent's defensive ranking for a stat type.
    Lower rank = better defense (harder matchup).
    """
    if stat_type == 'passing':
        # Get all QBs' passing yards against this defense
        opp_games = all_stats['passing'][
            all_stats['passing']['opponent'] == opponent
        ]
        avg_allowed = opp_games.groupby('opponent')['passing_yards'].mean()
        # Rank defenses (1 = stingiest, 32 = most generous)
        defense_ranks = avg_allowed.rank(method='min')
        return defense_ranks.get(opponent, 16)  # Default to league average
    
    # Similar logic for rushing/receiving
    return 16  # Default middle rank

# Update feature dict in get_player_features():
features.update({
    'opponent_def_rank': get_opponent_defense_rank(opponent, stat_type, all_stats),
    'is_home': 1 if is_home else 0,
    'days_rest': calculate_days_rest(player_latest['game_date'])
})
```

**Model Retraining Required**: Yes
**Expected Impact**: +3-5% accuracy improvement

---

### 2. Weather Impact for Outdoor Props
**Impact**: Medium | **Complexity**: Low | **Timeline**: 1 day

Integrate weather data for outdoor games (wind, precipitation, temperature).

```python
# Create new file: player_props/weather.py

import requests
from datetime import datetime

def get_game_weather(stadium, game_date):
    """
    Fetch weather forecast for game location.
    Returns dict with wind_mph, temp_f, precipitation_chance.
    """
    # Use free weather API (e.g., OpenWeatherMap)
    # Stadium locations stored in DATA_DIR/stadium_locations.csv
    
    return {
        'wind_mph': 15.0,
        'temp_f': 42.0,
        'precipitation_pct': 30.0,
        'is_dome': False
    }

def adjust_for_weather(prob_over, prop_type, weather):
    """
    Adjust prediction probability based on weather conditions.
    """
    if weather['is_dome']:
        return prob_over  # No weather impact
    
    # High wind (>15mph) reduces passing accuracy
    if prop_type == 'passing_yards' and weather['wind_mph'] > 15:
        prob_over *= 0.85  # Reduce OVER confidence by 15%
    
    # Cold weather (<32°F) reduces passing efficiency
    if prop_type in ['passing_yards', 'passing_tds'] and weather['temp_f'] < 32:
        prob_over *= 0.90
    
    # Rain increases rushing attempts
    if prop_type == 'rushing_yards' and weather['precipitation_pct'] > 60:
        prob_over *= 1.10
    
    return min(max(prob_over, 0.0), 1.0)  # Clamp to [0, 1]
```

**Integration Point**: `predict_props_for_game()` - apply adjustments after model prediction
**Data Source**: OpenWeatherMap API (free tier: 1000 calls/day)

---

### 3. Recent Form Weighting
**Impact**: High | **Complexity**: Medium | **Timeline**: 1 day

Weight recent games more heavily than older games in L3/L5/L10 calculations.

```python
# In player_props/aggregators.py - Update calculate_rolling_stats()

def calculate_weighted_rolling_stats(df, stat_cols, windows=[3, 5, 10]):
    """
    Calculate exponentially-weighted rolling stats.
    More recent games get higher weight.
    """
    df = df.sort_values('game_date')
    
    for window in windows:
        for col in stat_cols:
            # Exponential decay: alpha = 2/(window+1)
            df[f'{col}_L{window}'] = (
                df[col]
                .ewm(span=window, adjust=False)
                .mean()
                .shift(1)  # Exclude current game
            )
    
    return df

# Example: Last 5 games with exponential weighting
# Game 5 (oldest): weight = 0.10
# Game 4:          weight = 0.15
# Game 3:          weight = 0.20
# Game 2:          weight = 0.25
# Game 1 (newest): weight = 0.30
```

**Model Retraining Required**: Yes
**Expected Impact**: +2-4% accuracy for volatile players

---

### 4. Injury Status Integration
**Impact**: High | **Complexity**: High | **Timeline**: 2-3 days

Flag players with questionable/doubtful status and adjust predictions.

```python
# Create new file: player_props/injuries.py

import pandas as pd
from pathlib import Path

def load_injury_report():
    """
    Load latest NFL injury report from ESPN or NFL.com.
    Scrape weekly injury reports or use API.
    """
    # Mock structure - replace with actual scraping/API
    injuries = pd.DataFrame({
        'player_name': ['J.Cook', 'S.Barkley'],
        'status': ['Questionable', 'Probable'],
        'injury_type': ['Ankle', 'Rest'],
        'practice_participation': [
            'Limited',  # DNP/Limited/Full
            'Full'
        ]
    })
    return injuries

def adjust_for_injury(prediction, injury_status):
    """
    Adjust prediction based on injury status.
    """
    if injury_status is None:
        return prediction
    
    status = injury_status['status']
    
    # Questionable players: reduce OVER confidence by 20%
    if status == 'Questionable' and prediction['recommendation'] == 'OVER':
        prediction['confidence'] *= 0.80
        prediction['prob_over'] *= 0.80
        prediction['notes'] = f"⚠️ {injury_status['injury_type']} injury (Questionable)"
    
    # Out players: remove prediction entirely
    elif status == 'Out':
        return None
    
    return prediction

# In predict_props_for_game(), after generating predictions:
injuries = load_injury_report()
adjusted_predictions = []
for pred in predictions:
    injury = injuries[injuries['player_name'] == pred['player_name']]
    if not injury.empty:
        pred = adjust_for_injury(pred, injury.iloc[0])
    if pred is not None:
        adjusted_predictions.append(pred)
```

**Data Source**: ESPN injury reports (scraping) or Sportsdata.io API
**UI Enhancement**: Add injury icon/tooltip in Player Props table

---

## 🚀 Medium Priority Enhancements

### 5. Add Receptions Prop Type
**Impact**: Medium | **Complexity**: Low | **Timeline**: 1 day

Expand prop coverage with receptions for RBs/WRs/TEs.

```python
# In player_props/models.py - Add to PROP_LINES

PROP_LINES = {
    # ... existing props ...
    'receptions': {
        'elite_wr': 7.5,   # Elite WRs: 8+ receptions
        'star_wr': 5.5,    # Star WRs: 6+ receptions
        'good_wr': 4.5,    # Good WRs: 5+ receptions
        'starter': 3.5     # Role players: 4+ receptions
    }
}

# Update POSITION_MAP
POSITION_MAP = {
    'QB': ['passing_yards', 'passing_tds'],
    'RB': ['rushing_yards', 'rushing_tds', 'receptions'],  # Add receptions
    'WR': ['receiving_yards', 'receiving_tds', 'receptions'],
    'TE': ['receiving_yards', 'receiving_tds', 'receptions']
}

# Add training logic in train_all_models():
def train_receptions_models(stats_df):
    """Train models for receptions props."""
    models_trained = []
    
    for tier, line in PROP_LINES['receptions'].items():
        # Filter to players in this tier
        tier_df = filter_by_tier(stats_df, 'receptions', tier)
        
        # Create binary target
        tier_df['over_line'] = (tier_df['receptions'] > line).astype(int)
        
        # Train model
        model = train_xgboost_model(
            tier_df,
            features=['receptions_L3', 'receptions_L5', 'targets_L3', 'targets_L5'],
            target='over_line'
        )
        
        models_trained.append(f'receptions_{tier}')
    
    return models_trained
```

**New Features Required**:
- `receptions_L3`, `receptions_L5`, `receptions_L10` (already in receiving stats)
- `targets_L3`, `targets_L5` (already available)

---

### 6. Parlay Builder UI
**Impact**: Medium | **Complexity**: Medium | **Timeline**: 2 days

Help users build optimal parlays with correlation warnings.

```python
# Create new page: pages/3_🎲_Parlay_Builder.py

import streamlit as st
import pandas as pd
from itertools import combinations

st.title("🎲 Parlay Builder")

# Load predictions
predictions_df = load_player_props_predictions()

# Multiselect for parlay legs
st.subheader("Build Your Parlay")
selected_legs = st.multiselect(
    "Select props to combine (2-6 legs)",
    options=predictions_df.apply(
        lambda x: f"{x['display_name']} - {x['prop_type']} {x['recommendation']} {x['line_value']}", 
        axis=1
    ),
    max_selections=6
)

if len(selected_legs) >= 2:
    # Get selected rows
    leg_indices = [predictions_df.index[predictions_df.apply(
        lambda x: f"{x['display_name']} - {x['prop_type']} {x['recommendation']} {x['line_value']}" == leg,
        axis=1
    )][0] for leg in selected_legs]
    
    selected_props = predictions_df.loc[leg_indices]
    
    # Calculate parlay probability (assuming independence)
    combined_prob = selected_props['confidence'].prod()
    
    # Detect correlations
    correlations = detect_correlations(selected_props)
    
    # Display parlay summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Combined Probability", f"{combined_prob:.1%}")
    with col2:
        # American odds calculation
        decimal_odds = 1 / combined_prob
        american_odds = (decimal_odds - 1) * 100 if decimal_odds >= 2 else -100 / (decimal_odds - 1)
        st.metric("Fair Odds", f"+{american_odds:.0f}" if american_odds > 0 else f"{american_odds:.0f}")
    with col3:
        st.metric("Legs", len(selected_legs))
    
    # Show correlation warnings
    if correlations:
        st.warning("⚠️ Correlation Detected")
        for corr in correlations:
            st.write(f"- {corr['warning']}")
    
    # Parlay table
    st.dataframe(
        selected_props[['display_name', 'team', 'prop_type', 'recommendation', 'line_value', 'confidence']],
        use_container_width=True
    )

def detect_correlations(props_df):
    """
    Detect correlated props that may not be independent.
    """
    warnings = []
    
    # Check for same-game correlations
    for game in props_df.groupby(['team', 'opponent']):
        game_props = game[1]
        if len(game_props) > 1:
            # QB passing yards + team total OVERs = correlated
            qb_pass = game_props[game_props['prop_type'] == 'passing_yards']
            if not qb_pass.empty and qb_pass.iloc[0]['recommendation'] == 'OVER':
                warnings.append({
                    'warning': f"QB {qb_pass.iloc[0]['display_name']} passing OVER may correlate with other OVERs in this game"
                })
    
    # Check for opposing players (zero-sum correlations)
    teams = props_df['team'].unique()
    if len(teams) == 2 and props_df['opponent'].nunique() == 2:
        # Opposing QBs, RBs, etc. in same game
        warnings.append({
            'warning': "Legs contain opposing teams - outcomes may be negatively correlated"
        })
    
    return warnings
```

**Features**:
- Visual parlay builder with drag-and-drop
- Real-time probability calculations
- Correlation detection (same-game props, opposing players)
- Fair odds calculator

---

### 7. Player Performance Trends
**Impact**: Medium | **Complexity**: Low | **Timeline**: 1 day

Show trending indicators (↑ hot, ↓ cold, → stable).

```python
# In player_props/predict.py - Add to prediction dict

def calculate_trend(player_stats, stat_col):
    """
    Calculate if player is trending up, down, or stable.
    Returns: '↑' (hot), '→' (stable), '↓' (cold)
    """
    if len(player_stats) < 5:
        return '→'
    
    recent = player_stats.head(3)[stat_col].mean()  # Last 3 games
    older = player_stats.iloc[3:6][stat_col].mean()  # Games 4-6
    
    pct_change = (recent - older) / older if older > 0 else 0
    
    if pct_change > 0.20:
        return '🔥'  # Hot streak (20%+ increase)
    elif pct_change < -0.20:
        return '❄️'  # Cold streak (20%+ decrease)
    else:
        return '➡️'  # Stable

# Add to prediction dict:
prediction['trend'] = calculate_trend(player_stats, stat_col)
prediction['trend_pct'] = calculate_trend_percentage(player_stats, stat_col)
```

**UI Enhancement**:
```python
# In pages/2_🎯_Player_Props.py - Update column config

column_config={
    'trend': st.column_config.TextColumn('Trend', help='Recent performance trend'),
    # ... other columns ...
}
```

---

## 📊 Analytics & Monitoring

### 8. Prediction Accuracy Tracking
**Impact**: High | **Complexity**: Medium | **Timeline**: 2 days

Track actual results vs. predictions to calculate model accuracy.

```python
# Create new file: player_props/backtest.py

import pandas as pd
from datetime import datetime, timedelta

def collect_actual_results(week):
    """
    Fetch actual player stats for completed games.
    Use nfl_data_py or ESPN API.
    """
    import nfl_data_py as nfl
    
    # Get actual player stats for the week
    actual_stats = nfl.import_weekly_data([2025], columns=[
        'player_name', 'week', 'passing_yards', 'pass_tds',
        'rushing_yards', 'rush_tds', 'receiving_yards', 'rec_tds'
    ])
    
    return actual_stats[actual_stats['week'] == week]

def calculate_hit_rate(predictions_df, actuals_df):
    """
    Calculate prediction accuracy (hit rate).
    """
    merged = predictions_df.merge(
        actuals_df,
        on=['player_name', 'week'],
        suffixes=('_pred', '_actual')
    )
    
    results = []
    for _, row in merged.iterrows():
        prop_type = row['prop_type']
        stat_col = prop_type.replace('_', '')  # 'passing_yards' -> 'passingyards'
        
        actual_value = row[stat_col]
        line_value = row['line_value']
        predicted_rec = row['recommendation']
        
        # Determine actual outcome
        actual_outcome = 'OVER' if actual_value > line_value else 'UNDER'
        
        # Check if prediction was correct
        hit = (predicted_rec == actual_outcome)
        
        results.append({
            'player': row['player_name'],
            'prop': prop_type,
            'predicted': predicted_rec,
            'actual': actual_outcome,
            'hit': hit,
            'confidence': row['confidence']
        })
    
    results_df = pd.DataFrame(results)
    
    # Calculate metrics
    overall_hit_rate = results_df['hit'].mean()
    by_confidence = results_df.groupby(
        pd.cut(results_df['confidence'], bins=[0.55, 0.65, 0.75, 1.0])
    )['hit'].mean()
    
    return {
        'overall_accuracy': overall_hit_rate,
        'by_confidence_tier': by_confidence,
        'total_predictions': len(results_df),
        'detailed_results': results_df
    }

# Example usage:
if __name__ == '__main__':
    # Load last week's predictions
    predictions = pd.read_csv('data_files/player_props_predictions_week18.csv')
    
    # Get actual results
    actuals = collect_actual_results(week=18)
    
    # Calculate accuracy
    metrics = calculate_hit_rate(predictions, actuals)
    
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.1%}")
    print("\nBy Confidence Tier:")
    print(metrics['by_confidence_tier'])
```

**UI Page**: Create `pages/4_📈_Model_Performance.py` to display:
- Week-over-week accuracy trends
- Accuracy by prop type
- Accuracy by confidence tier
- Calibration curves (predicted prob vs actual outcome rate)

---

### 9. ROI Tracker
**Impact**: Medium | **Complexity**: Medium | **Timeline**: 2 days

Track hypothetical betting ROI to validate model profitability.

```python
# Create new file: player_props/roi_tracker.py

def calculate_roi(results_df, odds=-110):
    """
    Calculate ROI assuming standard -110 odds (bet $110 to win $100).
    """
    total_bets = len(results_df)
    total_hits = results_df['hit'].sum()
    
    # Calculate P&L
    # Each bet: risk $110, win $100 (or lose $110)
    total_wagered = total_bets * 110
    total_won = total_hits * 100
    total_lost = (total_bets - total_hits) * 110
    
    net_profit = total_won - total_lost
    roi = (net_profit / total_wagered) * 100
    
    return {
        'total_bets': total_bets,
        'wins': total_hits,
        'losses': total_bets - total_hits,
        'hit_rate': total_hits / total_bets,
        'total_wagered': total_wagered,
        'net_profit': net_profit,
        'roi': roi,
        'breakeven_rate': 52.4  # Need 52.4% to break even at -110
    }

# Filter to high-confidence bets only
def profitable_subset(results_df, min_confidence=0.65):
    """
    Calculate ROI for different confidence thresholds.
    """
    thresholds = [0.55, 0.60, 0.65, 0.70, 0.75]
    roi_by_threshold = {}
    
    for threshold in thresholds:
        subset = results_df[results_df['confidence'] >= threshold]
        if len(subset) > 0:
            roi_by_threshold[threshold] = calculate_roi(subset)
    
    return pd.DataFrame(roi_by_threshold).T
```

**UI Display**:
```python
# In pages/4_📈_Model_Performance.py

st.subheader("💰 ROI Analysis")

roi_table = profitable_subset(results_df)

st.dataframe(
    roi_table[['total_bets', 'hit_rate', 'roi']],
    column_config={
        'hit_rate': st.column_config.NumberColumn('Hit Rate', format='%.1f%%'),
        'roi': st.column_config.NumberColumn('ROI', format='%.1f%%')
    }
)

st.info(f"✅ Best Strategy: Bet on {roi_table['roi'].idxmax():.0%}+ confidence props")
```

---

## 🔮 Future Enhancements

### 10. Player Usage Trends
**Complexity**: High | **Timeline**: 3-4 days

Track snap count %, target share %, and touch share trends.

```python
# Requires additional data source
# nfl_data_py provides snap counts and target shares

def calculate_usage_metrics(player_name, team, all_stats):
    """
    Calculate player's recent usage trends.
    """
    player_stats = all_stats[all_stats['player_name'] == player_name]
    
    return {
        'snap_share_L3': player_stats.head(3)['snap_share'].mean(),
        'target_share_L3': player_stats.head(3)['target_share'].mean(),
        'red_zone_touches_L3': player_stats.head(3)['rz_touches'].mean()
    }
```

### 11. Multi-Model Ensemble
**Complexity**: High | **Timeline**: 3-5 days

Combine XGBoost with LightGBM and Random Forest for better predictions.

```python
from sklearn.ensemble import VotingClassifier
import lightgbm as lgb

def create_ensemble_model(X_train, y_train):
    """
    Ensemble of XGBoost, LightGBM, and Random Forest.
    """
    xgb_model = xgb.XGBClassifier(max_depth=4, learning_rate=0.1)
    lgb_model = lgb.LGBMClassifier(max_depth=4, learning_rate=0.1)
    
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', xgb_model),
            ('lgb', lgb_model)
        ],
        voting='soft',  # Use probability averaging
        weights=[0.6, 0.4]  # XGBoost slightly higher weight
    )
    
    ensemble.fit(X_train, y_train)
    return ensemble
```

### 12. Live In-Game Adjustments
**Complexity**: Very High | **Timeline**: 1-2 weeks

Adjust prop predictions based on in-game pace and script.

```python
def adjust_for_game_script(prediction, live_game_data):
    """
    Adjust predictions based on live game situation.
    
    Args:
        live_game_data: {
            'score_diff': -14,  # Team is down 14
            'time_remaining': 900,  # 15 minutes
            'game_script': 'pass_heavy'  # Or 'run_heavy', 'balanced'
        }
    """
    # Team down big -> more passing, less rushing
    if live_game_data['score_diff'] < -10:
        if prediction['prop_type'] == 'passing_yards':
            prediction['prob_over'] *= 1.15
        elif prediction['prop_type'] == 'rushing_yards':
            prediction['prob_over'] *= 0.85
    
    return prediction
```

---

## 📋 Implementation Priority

### Phase 1 (Week 1-2): Core Accuracy Improvements
1. ✅ Fix duplication bug
2. Opponent defense rankings
3. Recent form weighting
4. Weather impact

### Phase 2 (Week 3-4): Expand Coverage
5. Receptions prop type
6. Injury status integration
7. Player trends display

### Phase 3 (Week 5-6): Analytics & Tools
8. Prediction accuracy tracking
9. ROI tracker
10. Parlay builder UI

### Phase 4 (Future): Advanced Features
11. Player usage trends
12. Multi-model ensemble
13. Live in-game adjustments

---

## 🎯 Success Metrics

- **Accuracy**: Target 60%+ hit rate on high-confidence picks (≥65%)
- **ROI**: Target +5% ROI at -110 odds (52.4% breakeven)
- **Coverage**: 150+ props per week across 6 prop types
- **User Engagement**: Track parlay builder usage, downloads

---

## 📝 Notes

- All improvements maintain the **no data leakage** principle
- Model retraining required for feature additions
- UI changes require Streamlit app updates
- Consider Streamlit Cloud memory limits for large datasets

**Last Updated**: January 7, 2026
**Author**: GitHub Copilot
**Status**: Roadmap Draft
