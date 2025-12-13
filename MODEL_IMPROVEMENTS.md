# 🔬 Model Reliability Improvements

This document outlines suggested improvements to enhance model reliability, accuracy, and robustness for the NFL Predictions system.

**Last Updated**: December 12, 2025  
**Priority**: HIGH - These changes address core model performance and data quality issues

---

## 📋 Table of Contents

1. [Spread Betting Threshold Analysis](#0-spread-betting-threshold-analysis) ⚠️ **CRITICAL**
2. [Data Quality & Validation](#1-data-quality--validation)
3. [Feature Engineering Enhancements](#2-feature-engineering-enhancements)
4. [Model Architecture Improvements](#3-model-architecture-improvements)
5. [Calibration & Probability Refinement](#4-calibration--probability-refinement)
6. [Cross-Validation & Testing](#5-cross-validation--testing)
7. [Production Monitoring](#6-production-monitoring)

---

## 0. Spread Betting Threshold Analysis

### ⚠️ CRITICAL ISSUE: No Spread Bets Triggering

**Problem**: The current F1-optimized threshold for spread betting is too conservative, resulting in **zero games** meeting the threshold criteria. This defeats the purpose of having a spread betting model.

**Current State**:
- Spread threshold is optimized using F1-score on validation set
- F1 optimization tends to find very high thresholds to maximize precision
- Result: `pred_spreadCovered_optimal` == 1 for almost no games
- Users see "No spread bets available" for entire season

**Root Causes**:
1. **F1 Optimization Bias**: F1 score heavily penalizes false positives when positive class is rare, pushing threshold very high
2. **Class Imbalance**: Underdogs covering the spread is inherently close to 50/50, but F1 optimization treats it as imbalanced
3. **Inappropriate Metric**: F1 is designed for imbalanced classification (e.g., fraud detection), not betting where we want actionable predictions

---

### Solution 1: Lower Threshold to 50% (Simple & Immediate)

**Rationale**: For spread betting, we should bet when model is >50% confident the underdog covers. This is the natural threshold for binary classification.

**Implementation**:

```python
# In nfl-gather-data.py - Replace F1 optimization with fixed threshold

# OLD CODE (F1-optimized):
# thresholds = np.arange(0.1, 0.6, 0.02)
# f1_scores_spread = []
# for threshold in thresholds:
#     y_pred_thresh = (y_spread_proba >= threshold).astype(int)
#     f1 = f1_score(y_spread_test, y_pred_thresh, zero_division=0)
#     f1_scores_spread.append(f1)
# best_spread_threshold = thresholds[np.argmax(f1_scores_spread)]
# optimal_spread_threshold = best_spread_threshold

# NEW CODE (Fixed 50% threshold):
optimal_spread_threshold = 0.50

print(f"\n📊 Using fixed 50% threshold for spread betting (natural decision boundary)")

# Validate threshold makes sense
spread_bets_triggered = (y_spread_proba >= optimal_spread_threshold).sum()
print(f"   - {spread_bets_triggered}/{len(y_spread_proba)} test games trigger spread bet")
print(f"   - Expected ~{spread_bets_triggered/len(y_spread_proba)*100:.1f}% of games will have spread recommendations")
```

**Pros**:
- ✅ Immediate fix - no model retraining needed
- ✅ Mathematically sound for balanced classification
- ✅ Generates actionable predictions
- ✅ Easy to explain to users ("bet when >50% confident")

**Cons**:
- ⚠️ May generate more bets than optimal for bankroll management
- ⚠️ Doesn't account for edge/value (see Solution 2)

**Expected Outcome**: 40-50% of games will trigger spread betting signal

**Priority**: **CRITICAL** - Implement immediately  
**Effort**: 15 minutes (single line change)  
**Impact**: Restores spread betting functionality

---

### Solution 2: Expected Value (EV) Based Threshold (Recommended)

**Rationale**: Only bet when we have positive expected value against the implied odds. This is proper sports betting strategy.

**Implementation**:

```python
# In nfl-gather-data.py - Replace threshold optimization with EV-based approach

def calculate_spread_ev_threshold(model, X_test, y_test, spread_lines, min_edge=0.02):
    """
    Calculate EV-based betting threshold for spread bets.
    
    Args:
        model: Trained spread model
        X_test: Test features
        y_test: Test labels (1 = underdog covered)
        spread_lines: Array of spread lines for test games
        min_edge: Minimum edge required to bet (default 2%)
        
    Returns:
        Optimal threshold based on EV, and EV analysis
    """
    # Get model probabilities
    probs = model.predict_proba(X_test)[:, 1]
    
    # Standard spread odds are -110 (implied prob = 52.38%)
    implied_prob = 0.5238
    
    # Calculate edge for each prediction
    # Edge = Model Probability - Implied Probability
    edges = probs - implied_prob
    
    # Only bet when edge > min_edge AND probability > 50%
    ev_based_bets = (edges >= min_edge) & (probs >= 0.50)
    
    # Find threshold that captures these bets
    # The threshold is the minimum probability where we have sufficient edge
    if ev_based_bets.sum() > 0:
        optimal_threshold = probs[ev_based_bets].min()
    else:
        # Fallback to 50% if no +EV bets found
        optimal_threshold = 0.50
        print("⚠️ Warning: No +EV spread bets found in validation set")
    
    # Analysis
    print(f"\n📊 EV-Based Spread Threshold Analysis:")
    print(f"   - Optimal threshold: {optimal_threshold:.3f}")
    print(f"   - Min edge required: {min_edge*100:.1f}%")
    print(f"   - Bets triggered: {ev_based_bets.sum()}/{len(probs)} ({ev_based_bets.sum()/len(probs)*100:.1f}%)")
    
    # Backtest performance
    if ev_based_bets.sum() > 0:
        accuracy = y_test[ev_based_bets].mean()
        print(f"   - Historical accuracy on +EV bets: {accuracy*100:.1f}%")
        
        # Calculate theoretical ROI
        # Win: +$100 at -110 odds = $90.91 profit
        # Loss: -$110
        wins = y_test[ev_based_bets].sum()
        losses = ev_based_bets.sum() - wins
        profit = (wins * 90.91) - (losses * 110)
        roi = (profit / (ev_based_bets.sum() * 110)) * 100
        print(f"   - Theoretical ROI: {roi:.2f}%")
    
    return optimal_threshold, {
        'threshold': float(optimal_threshold),
        'min_edge': float(min_edge),
        'bets_triggered': int(ev_based_bets.sum()),
        'total_games': int(len(probs)),
        'bet_percentage': float(ev_based_bets.sum()/len(probs)*100)
    }


# Use in pipeline (after training spread model)
if __name__ == "__main__":
    # Replace F1 optimization with EV-based threshold
    optimal_spread_threshold, spread_ev_analysis = calculate_spread_ev_threshold(
        model_spread, 
        X_test_spread, 
        y_spread_test, 
        X_test_spread['spread_line'],  # Pass actual spread lines
        min_edge=0.02  # Require 2% edge minimum
    )
    
    # Save EV analysis to metrics
    metrics['Spread_EV_Analysis'] = spread_ev_analysis
```

**Pros**:
- ✅ Proper betting theory (only bet with positive EV)
- ✅ Controls for sportsbook's built-in edge (vig)
- ✅ Adjustable risk tolerance (min_edge parameter)
- ✅ Better bankroll preservation

**Cons**:
- ⚠️ More complex implementation
- ⚠️ May still generate fewer bets than desired if model isn't well-calibrated

**Expected Outcome**: 15-30% of games trigger spread betting (higher quality bets)

**Priority**: **HIGH** - Implement after Solution 1  
**Effort**: 2-3 hours  
**Impact**: Significantly improves betting strategy

---

### Solution 3: Multi-Tier Confidence System (User-Friendly)

**Rationale**: Let users choose their risk tolerance by showing multiple confidence tiers instead of binary yes/no.

**Implementation**:

```python
# In predictions.py - Add confidence tiers for spread bets

def add_spread_confidence_tiers(df):
    """
    Add confidence tier labels for spread betting.
    
    Tiers:
    - 🔥 Elite (≥60%): Highest confidence bets
    - ⭐ Strong (55-60%): Strong conviction bets
    - 📈 Good (52-55%): Positive edge bets
    - ⚖️ Lean (50-52%): Slight edge, lower units
    """
    df = df.copy()
    
    # Create tier column
    conditions = [
        df['prob_underdogCovered'] >= 0.60,
        (df['prob_underdogCovered'] >= 0.55) & (df['prob_underdogCovered'] < 0.60),
        (df['prob_underdogCovered'] >= 0.52) & (df['prob_underdogCovered'] < 0.55),
        (df['prob_underdogCovered'] >= 0.50) & (df['prob_underdogCovered'] < 0.52),
    ]
    
    choices = ['🔥 Elite', '⭐ Strong', '📈 Good', '⚖️ Lean']
    
    df['spread_confidence_tier'] = np.select(conditions, choices, default='')
    
    # Add recommended bet sizing
    # Elite: 3-5% of bankroll
    # Strong: 2-3% of bankroll
    # Good: 1-2% of bankroll
    # Lean: 0.5-1% of bankroll
    
    unit_conditions = [
        df['prob_underdogCovered'] >= 0.60,
        (df['prob_underdogCovered'] >= 0.55) & (df['prob_underdogCovered'] < 0.60),
        (df['prob_underdogCovered'] >= 0.52) & (df['prob_underdogCovered'] < 0.55),
        (df['prob_underdogCovered'] >= 0.50) & (df['prob_underdogCovered'] < 0.52),
    ]
    
    unit_choices = ['3-5%', '2-3%', '1-2%', '0.5-1%']
    
    df['recommended_bet_size'] = np.select(unit_conditions, unit_choices, default='')
    
    return df


# In main app - show tiered spread bets
if st.checkbox("Show Spread Bet Tiers", value=True):
    spread_bets_all = add_spread_confidence_tiers(predictions_df)
    
    # Filter to games with any tier (50%+)
    spread_bets_all = spread_bets_all[spread_bets_all['spread_confidence_tier'] != '']
    
    if len(spread_bets_all) > 0:
        st.write(f"### 🎯 Spread Betting Opportunities ({len(spread_bets_all)} games)")
        
        # Group by tier
        for tier in ['🔥 Elite', '⭐ Strong', '📈 Good', '⚖️ Lean']:
            tier_games = spread_bets_all[spread_bets_all['spread_confidence_tier'] == tier]
            
            if len(tier_games) > 0:
                with st.expander(f"{tier} ({len(tier_games)} games)", expanded=(tier in ['🔥 Elite', '⭐ Strong'])):
                    st.dataframe(
                        tier_games[['matchup', 'prob_underdogCovered', 'recommended_bet_size']],
                        hide_index=True
                    )
```

**Pros**:
- ✅ User-friendly - shows all opportunities with risk levels
- ✅ Flexible - users choose their risk tolerance
- ✅ Educational - teaches proper bankroll management
- ✅ Works with current model (no retraining)

**Cons**:
- ⚠️ More complex UI
- ⚠️ Requires user education

**Expected Outcome**: 40-50% of games show some tier, but users focus on Elite/Strong

**Priority**: **MEDIUM** - Good UX enhancement  
**Effort**: 3-4 hours  
**Impact**: Better user experience and education

---

### Solution 4: Improve Model to Get >50% Confidence (Long-term)

**Rationale**: The fundamental issue may be that the spread model isn't confident enough. Better features and model architecture could push probabilities higher.

**Implementation** (See other sections for details):

1. **Add Momentum Features** (Section 2.1)
   - Recent form, winning/losing streaks
   - Expected impact: +1-2% accuracy, +5-10% confidence on predictions

2. **Add Rest/Scheduling Features** (Section 2.2)
   - Thursday games, bye weeks, travel distance
   - Expected impact: +0.5-1% accuracy, especially for short-rest games

3. **Ensemble Model** (Section 3.1)
   - Stack XGBoost + Random Forest + Gradient Boosting
   - Expected impact: +2-3% accuracy, +10-15% confidence on predictions

4. **Better Calibration** (Section 4.1)
   - Isotonic regression is good, but can be improved
   - Expected impact: More reliable probability estimates

**Expected Timeline**: 2-3 weeks of development + testing

**Expected Outcome**: Model generates 60%+ probabilities for 10-20% of games

**Priority**: **HIGH** - Parallel track with Solution 1/2  
**Effort**: 20-30 hours  
**Impact**: Long-term sustainable improvement

---

### Recommended Implementation Plan

**Phase 1: Immediate (This Week)**
1. ✅ Implement Solution 1 (50% threshold) - 15 minutes
2. ✅ Deploy and monitor for 1-2 days
3. ✅ Collect user feedback

**Phase 2: Short-term (Next Week)**
1. ⏳ Implement Solution 2 (EV-based threshold) - 2-3 hours
2. ⏳ A/B test against 50% threshold
3. ⏳ Implement Solution 3 (confidence tiers) if users want more options

**Phase 3: Long-term (Next Month)**
1. ⏳ Implement Solution 4 (model improvements)
2. ⏳ Retrain and validate with new features
3. ⏳ Monitor for higher confidence predictions

---

### Expected Outcomes by Solution

| Solution | Bets Triggered | Quality | User Experience | Implementation Time |
|----------|---------------|---------|-----------------|-------------------|
| 1. 50% Threshold | 40-50% of games | Medium | Simple | 15 min |
| 2. EV-Based | 15-30% of games | High | Complex | 2-3 hours |
| 3. Multi-Tier | 40-50% (user choice) | Varies | Excellent | 3-4 hours |
| 4. Model Improvements | 10-20% at ≥60% | Very High | Best | 20-30 hours |

**Recommendation**: 
- **Immediate**: Implement Solution 1 (50% threshold) to restore functionality
- **This Week**: Add Solution 2 (EV-based) as default, keep 50% as fallback
- **This Month**: Implement Solution 4 (model improvements) to reduce reliance on threshold tuning

---

## 1. Data Quality & Validation

### 1.1 Missing Data Detection & Handling

**Problem**: Missing or null values in critical features can cause model failures or poor predictions.

**Implementation**:

```python
# In nfl-gather-data.py - Add data quality checks

def validate_feature_data(df, feature_list):
    """
    Validate that all required features have sufficient non-null values.
    
    Args:
        df: DataFrame with game features
        feature_list: List of feature names to validate
        
    Returns:
        Tuple of (is_valid, validation_report)
    """
    validation_report = {}
    is_valid = True
    
    for feature in feature_list:
        if feature not in df.columns:
            validation_report[feature] = {"status": "MISSING", "null_pct": 100.0}
            is_valid = False
            continue
            
        null_count = df[feature].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        
        if null_pct > 5.0:  # Flag if >5% missing
            validation_report[feature] = {
                "status": "HIGH_NULL_RATE",
                "null_pct": null_pct,
                "null_count": null_count
            }
            is_valid = False
        elif null_pct > 0:
            validation_report[feature] = {
                "status": "SOME_NULLS",
                "null_pct": null_pct,
                "null_count": null_count
            }
    
    return is_valid, validation_report


def impute_missing_features(df, feature_list):
    """
    Intelligently impute missing values based on feature type.
    
    Args:
        df: DataFrame with features
        feature_list: List of features to check
        
    Returns:
        DataFrame with imputed values
    """
    df = df.copy()
    
    for feature in feature_list:
        if feature not in df.columns:
            continue
            
        null_mask = df[feature].isnull()
        null_count = null_mask.sum()
        
        if null_count == 0:
            continue
            
        # For percentage/ratio features, use median
        if any(x in feature.lower() for x in ['pct', 'rate', 'ratio']):
            impute_value = df[feature].median()
        # For count/total features, use mean
        elif any(x in feature.lower() for x in ['avg', 'total', 'score']):
            impute_value = df[feature].mean()
        # For win/loss features, use conservative estimate
        elif 'win' in feature.lower():
            impute_value = 0.5  # Neutral win rate
        else:
            impute_value = df[feature].median()
        
        df.loc[null_mask, feature] = impute_value
        print(f"Imputed {null_count} missing values for {feature} with {impute_value:.4f}")
    
    return df


# Add to main data pipeline
if __name__ == "__main__":
    # After loading historical data
    historical_game_level_data = pd.read_csv(path.join(DATA_DIR, 'nfl_games_historical.csv'), sep='\t')
    
    # Validate data quality
    is_valid, validation_report = validate_feature_data(historical_game_level_data, features)
    
    if not is_valid:
        print("\n⚠️ DATA QUALITY ISSUES DETECTED:")
        for feature, report in validation_report.items():
            if report['status'] != 'OK':
                print(f"  - {feature}: {report['status']} ({report.get('null_pct', 0):.2f}% missing)")
        
        # Impute missing values
        print("\n🔧 Attempting to impute missing values...")
        historical_game_level_data = impute_missing_features(historical_game_level_data, features)
    
    # Save validation report
    with open(path.join(DATA_DIR, 'data_quality_report.json'), 'w') as f:
        json.dump(validation_report, f, indent=2)
```

**Priority**: HIGH  
**Effort**: 3-4 hours  
**Impact**: Prevents model failures and improves prediction reliability

---

### 1.2 Outlier Detection & Handling

**Problem**: Extreme outliers (e.g., 70-0 blowouts) can skew model training.

**Implementation**:

```python
# In nfl-gather-data.py

def detect_outliers(df, feature_list, method='iqr', threshold=3.0):
    """
    Detect outliers in numerical features using IQR or Z-score method.
    
    Args:
        df: DataFrame with features
        feature_list: List of numerical features to check
        method: 'iqr' or 'zscore'
        threshold: IQR multiplier (default 3.0) or Z-score threshold
        
    Returns:
        DataFrame with outlier flags added
    """
    df = df.copy()
    df['outlier_flags'] = 0
    
    for feature in feature_list:
        if feature not in df.columns:
            continue
            
        if method == 'iqr':
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = (df[feature] < lower_bound) | (df[feature] > upper_bound)
        
        elif method == 'zscore':
            mean = df[feature].mean()
            std = df[feature].std()
            z_scores = np.abs((df[feature] - mean) / std)
            outliers = z_scores > threshold
        
        if outliers.sum() > 0:
            print(f"Found {outliers.sum()} outliers in {feature}")
            df.loc[outliers, 'outlier_flags'] += 1
    
    return df


def handle_outliers(df, feature_list, method='winsorize', percentile=0.99):
    """
    Handle outliers by winsorizing (capping) at percentiles.
    
    Args:
        df: DataFrame with features
        feature_list: Features to process
        method: 'winsorize', 'remove', or 'cap'
        percentile: Upper percentile for winsorizing (e.g., 0.99 = 99th percentile)
        
    Returns:
        DataFrame with outliers handled
    """
    df = df.copy()
    
    if method == 'winsorize':
        for feature in feature_list:
            if feature not in df.columns:
                continue
            
            lower = df[feature].quantile(1 - percentile)
            upper = df[feature].quantile(percentile)
            
            original_min = df[feature].min()
            original_max = df[feature].max()
            
            df[feature] = df[feature].clip(lower=lower, upper=upper)
            
            if original_min < lower or original_max > upper:
                print(f"Winsorized {feature}: [{original_min:.2f}, {original_max:.2f}] -> [{lower:.2f}, {upper:.2f}]")
    
    elif method == 'remove':
        # Remove rows with multiple outlier flags
        outlier_threshold = len(feature_list) * 0.2  # If >20% of features are outliers
        df = df[df['outlier_flags'] < outlier_threshold]
        print(f"Removed {len(df[df['outlier_flags'] >= outlier_threshold])} rows with excessive outliers")
    
    return df


# Add to pipeline
if __name__ == "__main__":
    # After data validation
    historical_game_level_data = detect_outliers(historical_game_level_data, features)
    historical_game_level_data = handle_outliers(historical_game_level_data, features, method='winsorize')
```

**Priority**: MEDIUM  
**Effort**: 2-3 hours  
**Impact**: Reduces model sensitivity to extreme games

---

## 2. Feature Engineering Enhancements

### 2.1 Momentum & Streak Features

**Problem**: Current features don't capture team momentum (winning/losing streaks).

**Implementation**:

```python
# In nfl-gather-data.py - Add momentum features

def calculate_momentum_features(df):
    """
    Calculate team momentum and streak features.
    
    Args:
        df: DataFrame with game results sorted by date
        
    Returns:
        DataFrame with momentum features added
    """
    df = df.copy()
    df = df.sort_values(['season', 'team', 'gameday'])
    
    # Current winning/losing streak
    df['homeTeamCurrentStreak'] = 0
    df['awayTeamCurrentStreak'] = 0
    
    for team in df['home_team'].unique():
        team_games = df[df['home_team'] == team].index
        streak = 0
        
        for idx in team_games:
            if idx > 0:
                prev_result = df.loc[idx-1, 'homeTeamWon']
                if prev_result == 1:
                    streak = max(1, streak + 1)
                else:
                    streak = min(-1, streak - 1)
            
            df.loc[idx, 'homeTeamCurrentStreak'] = streak
    
    # Repeat for away team
    for team in df['away_team'].unique():
        team_games = df[df['away_team'] == team].index
        streak = 0
        
        for idx in team_games:
            if idx > 0:
                prev_result = df.loc[idx-1, 'awayTeamWon']
                if prev_result == 1:
                    streak = max(1, streak + 1)
                else:
                    streak = min(-1, streak - 1)
            
            df.loc[idx, 'awayTeamCurrentStreak'] = streak
    
    # Rolling 3-game momentum (weighted recent performance)
    df['homeTeamMomentum3'] = df.groupby('home_team')['homeTeamWon'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    df['awayTeamMomentum3'] = df.groupby('away_team')['awayTeamWon'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    
    # Momentum differential
    df['momentumDifferential'] = df['homeTeamMomentum3'] - df['awayTeamMomentum3']
    
    return df


# Add momentum features to feature list
momentum_features = [
    'homeTeamCurrentStreak', 'awayTeamCurrentStreak',
    'homeTeamMomentum3', 'awayTeamMomentum3',
    'momentumDifferential'
]

features.extend(momentum_features)
```

**Priority**: HIGH  
**Effort**: 4-5 hours  
**Impact**: Captures short-term team performance trends

---

### 2.2 Rest Days & Scheduling Features

**Problem**: Teams playing on short rest (Thursday games) perform differently.

**Implementation**:

```python
# In nfl-gather-data.py

def calculate_rest_features(df):
    """
    Calculate days of rest and scheduling features.
    
    Args:
        df: DataFrame with game dates
        
    Returns:
        DataFrame with rest features added
    """
    df = df.copy()
    df['gameday'] = pd.to_datetime(df['gameday'])
    df = df.sort_values(['team', 'gameday'])
    
    # Days since last game
    df['homeTeamDaysRest'] = df.groupby('home_team')['gameday'].diff().dt.days
    df['awayTeamDaysRest'] = df.groupby('away_team')['gameday'].diff().dt.days
    
    # Fill first game of season with 7 (normal rest)
    df['homeTeamDaysRest'].fillna(7, inplace=True)
    df['awayTeamDaysRest'].fillna(7, inplace=True)
    
    # Rest advantage (positive = home team more rested)
    df['restAdvantage'] = df['homeTeamDaysRest'] - df['awayTeamDaysRest']
    
    # Short rest indicator (less than 6 days)
    df['homeTeamShortRest'] = (df['homeTeamDaysRest'] < 6).astype(int)
    df['awayTeamShortRest'] = (df['awayTeamDaysRest'] < 6).astype(int)
    
    # Extra rest indicator (bye week = 14+ days)
    df['homeTeamExtraRest'] = (df['homeTeamDaysRest'] >= 14).astype(int)
    df['awayTeamExtraRest'] = (df['awayTeamDaysRest'] >= 14).astype(int)
    
    # Day of week (Thursday = 3, Sunday = 6, Monday = 0)
    df['dayOfWeek'] = df['gameday'].dt.dayofweek
    df['isThursdayGame'] = (df['dayOfWeek'] == 3).astype(int)
    df['isMondayGame'] = (df['dayOfWeek'] == 0).astype(int)
    
    return df


# Add rest features
rest_features = [
    'homeTeamDaysRest', 'awayTeamDaysRest', 'restAdvantage',
    'homeTeamShortRest', 'awayTeamShortRest',
    'homeTeamExtraRest', 'awayTeamExtraRest',
    'isThursdayGame', 'isMondayGame'
]

features.extend(rest_features)
```

**Priority**: MEDIUM  
**Effort**: 3-4 hours  
**Impact**: Accounts for fatigue and scheduling advantages

---

### 2.3 Weather & Environmental Features

**Problem**: Weather conditions (wind, rain, cold) significantly impact scoring.

**Implementation**:

```python
# In nfl-gather-data.py
# Note: Requires weather data integration (ESPN API or weather service)

def add_weather_features(df):
    """
    Add weather and environmental features to game data.
    
    Args:
        df: DataFrame with game information
        
    Returns:
        DataFrame with weather features added
    """
    # This is a placeholder - actual implementation requires weather API
    # Example: OpenWeatherMap API, ESPN weather data, or NFL.com
    
    df = df.copy()
    
    # Indoor/outdoor stadium indicator
    indoor_stadiums = [
        'ATL', 'DET', 'HOU', 'IND', 'LAC', 'LAR', 
        'LV', 'MIN', 'NO', 'ARI'
    ]
    
    df['isIndoorGame'] = df['home_team'].isin(indoor_stadiums).astype(int)
    
    # For outdoor games, add weather features (mock data - replace with API)
    df['temperature'] = 65.0  # Degrees F
    df['windSpeed'] = 5.0     # MPH
    df['precipitation'] = 0.0  # Inches
    df['isDomeGame'] = df['isIndoorGame']
    
    # Weather severity score (higher = worse conditions)
    df['weatherSeverity'] = 0.0
    df.loc[~df['isIndoorGame'], 'weatherSeverity'] = (
        (df['windSpeed'] / 20.0) * 0.5 +  # Wind impact
        (df['precipitation']) * 0.3 +       # Rain/snow impact
        (np.abs(df['temperature'] - 65) / 30.0) * 0.2  # Temperature extremes
    )
    
    return df


# Add weather features
weather_features = [
    'isIndoorGame', 'temperature', 'windSpeed', 
    'precipitation', 'weatherSeverity'
]

features.extend(weather_features)
```

**Priority**: LOW (requires external data source)  
**Effort**: 6-8 hours (including API integration)  
**Impact**: Improves over/under predictions significantly

---

## 3. Model Architecture Improvements

### 3.1 Ensemble Model with Stacking

**Problem**: Single XGBoost model may not capture all patterns.

**Implementation**:

```python
# In nfl-gather-data.py - Replace single model with ensemble

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

def create_ensemble_model(X_train, y_train, model_type='spread'):
    """
    Create an ensemble model with stacking.
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_type: 'spread', 'moneyline', or 'totals'
        
    Returns:
        Fitted stacking ensemble model
    """
    # Base models
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric='logloss'
    )
    
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    
    gb_model = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    
    # Meta-learner (uses base model predictions)
    meta_model = LogisticRegression(max_iter=1000, random_state=42)
    
    # Create stacking ensemble
    stacking_model = StackingClassifier(
        estimators=[
            ('xgb', xgb_model),
            ('rf', rf_model),
            ('gb', gb_model)
        ],
        final_estimator=meta_model,
        cv=5,
        stack_method='predict_proba'
    )
    
    print(f"Training {model_type} ensemble model...")
    stacking_model.fit(X_train, y_train)
    
    return stacking_model


# Replace single model training
if __name__ == "__main__":
    # Train ensemble models instead of single XGBoost
    spread_model = create_ensemble_model(X_train_spread, y_spread_train, 'spread')
    moneyline_model = create_ensemble_model(X_train_ml, y_train_ml, 'moneyline')
    totals_model = create_ensemble_model(X_train_totals, y_train_totals, 'totals')
    
    # Evaluate ensemble
    spread_probs = spread_model.predict_proba(X_test_spread)[:, 1]
    spread_accuracy = (spread_model.predict(X_test_spread) == y_spread_test).mean()
    print(f"Spread ensemble accuracy: {spread_accuracy:.4f}")
```

**Priority**: HIGH  
**Effort**: 6-8 hours  
**Impact**: Typically improves accuracy by 2-3% and reduces variance

---

### 3.2 Hyperparameter Optimization

**Problem**: Current hyperparameters are not optimized for this specific dataset.

**Implementation**:

```python
# In nfl-gather-data.py

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

def optimize_xgb_hyperparameters(X_train, y_train, n_iter=50):
    """
    Use RandomizedSearchCV to find optimal hyperparameters.
    
    Args:
        X_train: Training features
        y_train: Training labels
        n_iter: Number of parameter combinations to try
        
    Returns:
        Best estimator and parameters
    """
    # Parameter search space
    param_distributions = {
        'n_estimators': randint(100, 500),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.2),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'min_child_weight': randint(1, 10),
        'gamma': uniform(0, 0.5),
        'reg_alpha': uniform(0, 1),
        'reg_lambda': uniform(0, 2)
    }
    
    # Base model
    xgb_model = xgb.XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    # Randomized search with cross-validation
    random_search = RandomizedSearchCV(
        xgb_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=5,
        scoring='neg_log_loss',
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    print(f"Searching {n_iter} parameter combinations...")
    random_search.fit(X_train, y_train)
    
    print(f"\nBest parameters found:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")
    
    print(f"\nBest cross-validation score: {-random_search.best_score_:.4f}")
    
    return random_search.best_estimator_, random_search.best_params_


# Add to pipeline
if __name__ == "__main__":
    # Optimize each model separately
    spread_model, spread_params = optimize_xgb_hyperparameters(
        X_train_spread, y_spread_train, n_iter=100
    )
    
    # Save best parameters
    best_params = {
        'spread': spread_params,
        'moneyline': moneyline_params,
        'totals': totals_params
    }
    
    with open(path.join(DATA_DIR, 'best_hyperparameters.json'), 'w') as f:
        json.dump(best_params, f, indent=2)
```

**Priority**: MEDIUM  
**Effort**: 4-6 hours (mostly computation time)  
**Impact**: 1-2% accuracy improvement

---

## 4. Calibration & Probability Refinement

### 4.1 Improved Probability Calibration

**Problem**: Current isotonic regression may not be optimal for all probability ranges.

**Implementation**:

```python
# In nfl-gather-data.py

from sklearn.calibration import CalibratedClassifierCV, calibration_curve

def calibrate_model_advanced(model, X_val, y_val, method='sigmoid'):
    """
    Apply advanced probability calibration with diagnostic plots.
    
    Args:
        model: Trained classifier
        X_val: Validation features
        y_val: Validation labels
        method: 'sigmoid' or 'isotonic'
        
    Returns:
        Calibrated model and calibration report
    """
    # Get uncalibrated probabilities
    y_prob_uncal = model.predict_proba(X_val)[:, 1]
    
    # Apply calibration
    calibrated_model = CalibratedClassifierCV(
        model,
        method=method,
        cv='prefit'
    )
    calibrated_model.fit(X_val, y_val)
    
    # Get calibrated probabilities
    y_prob_cal = calibrated_model.predict_proba(X_val)[:, 1]
    
    # Calculate calibration curves
    fraction_positives_uncal, mean_predicted_uncal = calibration_curve(
        y_val, y_prob_uncal, n_bins=10
    )
    fraction_positives_cal, mean_predicted_cal = calibration_curve(
        y_val, y_prob_cal, n_bins=10
    )
    
    # Calculate calibration metrics
    from sklearn.metrics import brier_score_loss
    brier_uncal = brier_score_loss(y_val, y_prob_uncal)
    brier_cal = brier_score_loss(y_val, y_prob_cal)
    
    calibration_report = {
        'brier_score_uncalibrated': brier_uncal,
        'brier_score_calibrated': brier_cal,
        'improvement': (brier_uncal - brier_cal) / brier_uncal * 100,
        'calibration_method': method
    }
    
    print(f"\nCalibration Results:")
    print(f"  Brier Score (uncalibrated): {brier_uncal:.4f}")
    print(f"  Brier Score (calibrated): {brier_cal:.4f}")
    print(f"  Improvement: {calibration_report['improvement']:.2f}%")
    
    return calibrated_model, calibration_report


# Apply to all models
if __name__ == "__main__":
    # After training, calibrate on validation set
    spread_model_cal, spread_cal_report = calibrate_model_advanced(
        spread_model, X_test_spread, y_spread_test, method='isotonic'
    )
    
    # Save calibration reports
    calibration_reports = {
        'spread': spread_cal_report,
        'moneyline': moneyline_cal_report,
        'totals': totals_cal_report
    }
    
    with open(path.join(DATA_DIR, 'calibration_reports.json'), 'w') as f:
        json.dump(calibration_reports, f, indent=2)
```

**Priority**: HIGH  
**Effort**: 3-4 hours  
**Impact**: More reliable probability estimates for betting

---

## 5. Cross-Validation & Testing

### 5.1 Time-Series Cross-Validation

**Problem**: Standard random split doesn't respect temporal ordering of games.

**Implementation**:

```python
# In nfl-gather-data.py

from sklearn.model_selection import TimeSeriesSplit

def time_series_cv_evaluate(X, y, model, n_splits=5):
    """
    Perform time-series cross-validation to better simulate real prediction.
    
    Args:
        X: Features (must be sorted by date)
        y: Labels
        model: Model to evaluate
        n_splits: Number of CV splits
        
    Returns:
        Dictionary with CV scores and predictions
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    cv_scores = []
    cv_predictions = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
        y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train model on this fold
        model_copy = clone(model)
        model_copy.fit(X_train_cv, y_train_cv)
        
        # Evaluate
        y_pred = model_copy.predict(X_test_cv)
        y_prob = model_copy.predict_proba(X_test_cv)[:, 1]
        
        accuracy = (y_pred == y_test_cv).mean()
        cv_scores.append(accuracy)
        
        cv_predictions.append({
            'fold': fold,
            'test_indices': test_idx,
            'predictions': y_pred,
            'probabilities': y_prob,
            'actuals': y_test_cv.values
        })
        
        print(f"Fold {fold+1}/{n_splits}: Accuracy = {accuracy:.4f}")
    
    results = {
        'mean_accuracy': np.mean(cv_scores),
        'std_accuracy': np.std(cv_scores),
        'fold_scores': cv_scores,
        'predictions': cv_predictions
    }
    
    print(f"\nTime-Series CV Results:")
    print(f"  Mean Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
    
    return results


# Use in pipeline
if __name__ == "__main__":
    # Ensure data is sorted by date
    historical_game_level_data = historical_game_level_data.sort_values('gameday')
    
    # Evaluate with time-series CV
    spread_cv_results = time_series_cv_evaluate(
        X_spread_full, y_spread, spread_model, n_splits=5
    )
    
    # Save CV results
    with open(path.join(DATA_DIR, 'time_series_cv_results.json'), 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {k: v for k, v in spread_cv_results.items() 
                       if k != 'predictions'}
        json.dump(json_results, f, indent=2)
```

**Priority**: HIGH  
**Effort**: 4-5 hours  
**Impact**: More realistic performance estimates

---

### 5.2 Stratified Evaluation by Game Context

**Problem**: Model may perform differently for close games vs. blowouts.

**Implementation**:

```python
# In nfl-gather-data.py

def evaluate_by_context(y_true, y_pred, y_prob, context_feature):
    """
    Evaluate model performance stratified by game context.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        context_feature: Feature to stratify by (e.g., spread_line)
        
    Returns:
        Dictionary with performance by context
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
    
    # Define context bins
    if 'spread' in context_feature.name:
        bins = [-np.inf, -7, -3, 3, 7, np.inf]
        labels = ['Big Favorite', 'Favorite', 'Pick Em', 'Underdog', 'Big Underdog']
    elif 'total' in context_feature.name:
        bins = [0, 42, 47, 52, np.inf]
        labels = ['Low Scoring', 'Below Avg', 'Average', 'High Scoring']
    else:
        # Generic quartile binning
        bins = context_feature.quantile([0, 0.25, 0.5, 0.75, 1.0]).values
        labels = ['Q1', 'Q2', 'Q3', 'Q4']
    
    context_bins = pd.cut(context_feature, bins=bins, labels=labels)
    
    results = {}
    for context in labels:
        mask = context_bins == context
        if mask.sum() == 0:
            continue
        
        results[context] = {
            'count': mask.sum(),
            'accuracy': accuracy_score(y_true[mask], y_pred[mask]),
            'precision': precision_score(y_true[mask], y_pred[mask], zero_division=0),
            'recall': recall_score(y_true[mask], y_pred[mask], zero_division=0),
            'roc_auc': roc_auc_score(y_true[mask], y_prob[mask]) if mask.sum() > 1 else None
        }
    
    # Print summary
    print("\nPerformance by Context:")
    print(f"{'Context':<20} {'Count':<10} {'Accuracy':<12} {'ROC AUC':<12}")
    print("-" * 60)
    for context, metrics in results.items():
        auc_str = f"{metrics['roc_auc']:.4f}" if metrics['roc_auc'] else "N/A"
        print(f"{context:<20} {metrics['count']:<10} {metrics['accuracy']:<12.4f} {auc_str:<12}")
    
    return results


# Use in evaluation
if __name__ == "__main__":
    # After getting predictions
    spread_context_eval = evaluate_by_context(
        y_spread_test,
        spread_model.predict(X_test_spread),
        spread_model.predict_proba(X_test_spread)[:, 1],
        X_test_spread['spread_line']
    )
    
    # Save context evaluation
    with open(path.join(DATA_DIR, 'context_evaluation.json'), 'w') as f:
        json.dump(spread_context_eval, f, indent=2)
```

**Priority**: MEDIUM  
**Effort**: 3-4 hours  
**Impact**: Identifies where model needs improvement

---

## 6. Production Monitoring

### 6.1 Prediction Confidence Tracking

**Problem**: No visibility into when the model is uncertain.

**Implementation**:

```python
# In predictions.py

def analyze_prediction_confidence(predictions_df):
    """
    Analyze and flag low-confidence predictions.
    
    Args:
        predictions_df: DataFrame with model predictions
        
    Returns:
        DataFrame with confidence flags added
    """
    df = predictions_df.copy()
    
    # Calculate prediction uncertainty (entropy)
    for market in ['spread', 'moneyline', 'totals']:
        prob_col = f'prob_{market}'
        if prob_col in df.columns:
            # Entropy: -p*log(p) - (1-p)*log(1-p)
            p = df[prob_col].clip(0.01, 0.99)  # Avoid log(0)
            entropy = -(p * np.log2(p) + (1-p) * np.log2(1-p))
            df[f'{market}_uncertainty'] = entropy
            
            # Flag high uncertainty (entropy > 0.9 means close to 50/50)
            df[f'{market}_high_uncertainty'] = (entropy > 0.9).astype(int)
    
    # Flag predictions with conflicting signals
    if all(col in df.columns for col in ['prob_spread', 'prob_moneyline']):
        spread_says_favorite = df['prob_spread'] > 0.5
        ml_says_favorite = df['prob_moneyline'] > 0.5
        df['conflicting_predictions'] = (spread_says_favorite != ml_says_favorite).astype(int)
    
    # Summary statistics
    high_uncertainty_count = df[[c for c in df.columns if 'high_uncertainty' in c]].sum().sum()
    conflicting_count = df.get('conflicting_predictions', pd.Series([0])).sum()
    
    st.sidebar.write("### 🔍 Prediction Quality")
    st.sidebar.metric("High Uncertainty Games", high_uncertainty_count)
    if conflicting_count > 0:
        st.sidebar.warning(f"⚠️ {conflicting_count} games with conflicting predictions")
    
    return df


# Add to main predictions.py
if __name__ == "__main__":
    # After loading predictions
    predictions_df = analyze_prediction_confidence(predictions_df)
```

**Priority**: MEDIUM  
**Effort**: 2-3 hours  
**Impact**: Better transparency and risk management

---

### 6.2 Real-Time Performance Tracking

**Problem**: No automated tracking of prediction accuracy over time.

**Implementation**:

```python
# Create new file: scripts/track_performance.py

import pandas as pd
import numpy as np
from datetime import datetime
import os.path as path

DATA_DIR = 'data_files/'

def update_performance_log():
    """
    Fetch actual results and update performance tracking log.
    """
    # Load betting log
    log_path = path.join(DATA_DIR, 'betting_recommendations_log.csv')
    if not os.path.exists(log_path):
        print("No betting log found")
        return
    
    betting_log = pd.read_csv(log_path)
    
    # Load predictions
    predictions_path = path.join(DATA_DIR, 'nfl_games_historical_with_predictions.csv')
    predictions = pd.read_csv(predictions_path, sep='\t')
    
    # Calculate rolling performance metrics
    recent_games = betting_log[betting_log['bet_result'].isin(['win', 'loss'])]
    
    if len(recent_games) == 0:
        print("No completed bets to track")
        return
    
    # Overall metrics
    total_bets = len(recent_games)
    wins = (recent_games['bet_result'] == 'win').sum()
    win_rate = wins / total_bets if total_bets > 0 else 0
    
    # By confidence tier
    tier_performance = recent_games.groupby('confidence_tier').agg({
        'bet_result': lambda x: (x == 'win').mean(),
        'bet_profit': 'sum'
    }).to_dict()
    
    # Create performance snapshot
    snapshot = {
        'timestamp': datetime.now().isoformat(),
        'total_bets': total_bets,
        'wins': wins,
        'losses': total_bets - wins,
        'win_rate': win_rate,
        'total_profit': recent_games['bet_profit'].sum(),
        'roi': (recent_games['bet_profit'].sum() / (total_bets * 100)) if total_bets > 0 else 0,
        'tier_performance': tier_performance
    }
    
    # Append to performance log
    log_file = path.join(DATA_DIR, 'performance_tracking.jsonl')
    with open(log_file, 'a') as f:
        f.write(json.dumps(snapshot) + '\n')
    
    print(f"\n📊 Performance Update ({datetime.now().strftime('%Y-%m-%d %H:%M')})")
    print(f"Total Bets: {total_bets}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"ROI: {snapshot['roi']:.2%}")
    
    return snapshot


if __name__ == "__main__":
    update_performance_log()
```

**Priority**: LOW (nice-to-have for production)  
**Effort**: 3-4 hours  
**Impact**: Automated performance monitoring

---

## 🎯 Implementation Priority Roadmap

### Phase 1: Critical (Week 1-2)
1. ✅ Data Quality Validation & Imputation
2. ✅ Time-Series Cross-Validation
3. ✅ Improved Probability Calibration
4. ✅ Momentum & Streak Features

### Phase 2: High Value (Week 3-4)
5. ✅ Outlier Detection & Handling
6. ✅ Ensemble Model with Stacking
7. ✅ Stratified Context Evaluation
8. ✅ Rest Days & Scheduling Features

### Phase 3: Optimization (Week 5-6)
9. ⏳ Hyperparameter Optimization
10. ⏳ Prediction Confidence Tracking
11. ⏳ Weather Features (if API available)

### Phase 4: Production (Week 7-8)
12. ⏳ Real-Time Performance Tracking
13. ⏳ Automated retraining pipeline
14. ⏳ A/B testing framework

---

## 📊 Expected Impact Summary

| Improvement | Expected Accuracy Gain | Implementation Effort | Priority |
|------------|----------------------|---------------------|----------|
| **Spread Threshold Fix** | **Restores functionality** | **15 min** | **CRITICAL** |
| EV-Based Threshold | Better bet quality | 2-3 hours | HIGH |
| Multi-Tier Confidence | User experience | 3-4 hours | MEDIUM |
| Data Quality Validation | +0.5-1% | 3-4 hours | HIGH |
| Momentum Features | +1-2% | 4-5 hours | HIGH |
| Time-Series CV | Better estimates | 4-5 hours | HIGH |
| Ensemble Model | +2-3% | 6-8 hours | HIGH |
| Calibration Improvements | Better probabilities | 3-4 hours | HIGH |
| Rest/Scheduling Features | +0.5-1% | 3-4 hours | MEDIUM |
| Hyperparameter Tuning | +1-2% | 4-6 hours | MEDIUM |
| Outlier Handling | +0.5% | 2-3 hours | MEDIUM |
| Weather Features | +1-1.5% (totals) | 6-8 hours | LOW |
| Context Evaluation | Insights only | 3-4 hours | MEDIUM |

**Total Expected Improvement**: 5-10% accuracy gain across all markets

**Critical Note**: The spread threshold fix is **MANDATORY** - without it, the spread betting feature is non-functional.

---

## 🔧 Testing & Validation

After implementing each improvement:

1. **Backtesting**: Run on historical data (2020-2024 seasons)
2. **Cross-Validation**: Use time-series CV to validate improvement
3. **Context Analysis**: Check performance across different game contexts
4. **Calibration Check**: Verify probability estimates are well-calibrated
5. **A/B Comparison**: Compare new model vs. current model side-by-side

---

## 📝 Notes

- Prioritize data quality and feature engineering before complex model architectures
- Always validate improvements with proper cross-validation
- Monitor for overfitting when adding features
- Keep model interpretability in mind for betting decisions
- Document all changes in model_metrics.json

**Last Updated**: December 12, 2025
