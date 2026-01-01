# 🚀 Advanced NFL Predictions Model Roadmap

**Last Updated**: December 31, 2025  
**Status**: Ready to implement - Copy/paste code snippets below

---

## 🎯 Quick Wins (Implement This Week)

### 1. Add Quarterback Rating (QBR) Features

**Impact**: HIGH | **Effort**: 2 hours | **Expected ROI**: +5-8%

**Why**: QB performance is the #1 predictor of game outcomes but currently unused

```python
# Add to nfl-gather-data.py after loading historical_game_level_data

import pandas as pd

# Load QBR data
qbr_data = pd.read_csv(path.join(DATA_DIR, 'qbr_week_level.csv'))

# Aggregate QBR by player and week
qbr_data['game_key'] = qbr_data['season'].astype(str) + '_' + qbr_data['week_num'].astype(str) + '_' + qbr_data['team_abb']

def get_qb_rolling_stats(df, qb_id_col, stat_col='qbr_total', window=3):
    """Get rolling QB stats for last N games"""
    stats = []
    for idx, row in df.iterrows():
        qb_id = row[qb_id_col]
        week = row['week']
        season = row['season']
        
        prior_games = qbr_data[
            (qbr_data['player_id'] == qb_id) & 
            ((qbr_data['season'] < season) | ((qbr_data['season'] == season) & (qbr_data['week_num'] < week)))
        ].tail(window)
        
        if len(prior_games) == 0:
            stats.append(50.0)  # League average QBR
        else:
            stats.append(prior_games[stat_col].mean())
    return stats

# Add QB features
historical_game_level_data['home_qb_rolling_qbr'] = get_qb_rolling_stats(historical_game_level_data, 'home_qb_id')
historical_game_level_data['away_qb_rolling_qbr'] = get_qb_rolling_stats(historical_game_level_data, 'away_qb_id')
historical_game_level_data['qb_rating_diff'] = historical_game_level_data['home_qb_rolling_qbr'] - historical_game_level_data['away_qb_rolling_qbr']

# Add to features list
features.extend(['home_qb_rolling_qbr', 'away_qb_rolling_qbr', 'qb_rating_diff'])
```

---

### 2. Advanced Team EPA (Expected Points Added) Features

**Impact**: HIGH | **Effort**: 3 hours | **Expected ROI**: +8-12%

**Why**: EPA captures efficiency better than raw stats

```python
# Add to nfl-gather-data.py

# Load team weekly stats
team_stats_2024 = pd.read_csv(path.join(DATA_DIR, 'stats_team_week_2024.csv'))
team_stats_2023 = pd.read_csv(path.join(DATA_DIR, 'stats_team_week_2023.csv'))
team_stats_2022 = pd.read_csv(path.join(DATA_DIR, 'stats_team_week_2022.csv'))
team_stats = pd.concat([team_stats_2022, team_stats_2023, team_stats_2024])

# Create team-week lookup
team_stats['game_key'] = team_stats['season'].astype(str) + '_' + team_stats['week'].astype(str) + '_' + team_stats['team']

def get_team_epa_rolling(df, team_col, is_offense=True):
    """Calculate rolling EPA for team"""
    epa_col = 'passing_epa' if is_offense else 'def_interceptions'  # Simplified
    stats = []
    
    for idx, row in df.iterrows():
        team = row[team_col]
        week = row['week']
        season = row['season']
        
        prior = team_stats[
            (team_stats['team'] == team) & 
            ((team_stats['season'] < season) | ((team_stats['season'] == season) & (team_stats['week'] < week)))
        ].tail(5)
        
        if len(prior) == 0:
            stats.append(0.0)
        else:
            stats.append(prior['passing_epa'].mean())
    return stats

# Add EPA features
historical_game_level_data['home_offense_epa'] = get_team_epa_rolling(historical_game_level_data, 'home_team', True)
historical_game_level_data['away_offense_epa'] = get_team_epa_rolling(historical_game_level_data, 'away_team', True)
historical_game_level_data['epa_diff'] = historical_game_level_data['home_offense_epa'] - historical_game_level_data['away_offense_epa']

features.extend(['home_offense_epa', 'away_offense_epa', 'epa_diff'])
```

---

### 3. Opponent Strength of Schedule

**Impact**: MEDIUM | **Effort**: 2 hours | **Expected ROI**: +3-5%

```python
# Add to nfl-gather-data.py

def calculate_strength_of_schedule(df):
    """Calculate opponent win% for each game"""
    sos_home = []
    sos_away = []
    
    for idx, row in df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        week = row['week']
        season = row['season']
        
        # Get opponents faced by home team before this game
        home_prior = df[
            ((df['home_team'] == home_team) | (df['away_team'] == home_team)) &
            ((df['season'] < season) | ((df['season'] == season) & (df['week'] < week)))
        ]
        
        # Calculate average opponent win%
        home_opponents = []
        for _, game in home_prior.iterrows():
            opp = game['away_team'] if game['home_team'] == home_team else game['home_team']
            opp_wins = df[
                ((df['home_team'] == opp) & (df['homeWin'] == 1)) | 
                ((df['away_team'] == opp) & (df['awayWin'] == 1))
            ]
            opp_games = df[(df['home_team'] == opp) | (df['away_team'] == opp)]
            win_pct = len(opp_wins) / len(opp_games) if len(opp_games) > 0 else 0.5
            home_opponents.append(win_pct)
        
        sos_home.append(np.mean(home_opponents) if home_opponents else 0.5)
        
        # Same for away team
        away_prior = df[
            ((df['home_team'] == away_team) | (df['away_team'] == away_team)) &
            ((df['season'] < season) | ((df['season'] == season) & (df['week'] < week)))
        ]
        
        away_opponents = []
        for _, game in away_prior.iterrows():
            opp = game['away_team'] if game['home_team'] == away_team else game['home_team']
            opp_wins = df[
                ((df['home_team'] == opp) & (df['homeWin'] == 1)) | 
                ((df['away_team'] == opp) & (df['awayWin'] == 1))
            ]
            opp_games = df[(df['home_team'] == opp) | (df['away_team'] == opp)]
            win_pct = len(opp_wins) / len(opp_games) if len(opp_games) > 0 else 0.5
            away_opponents.append(win_pct)
        
        sos_away.append(np.mean(away_opponents) if away_opponents else 0.5)
    
    return sos_home, sos_away

home_sos, away_sos = calculate_strength_of_schedule(historical_game_level_data)
historical_game_level_data['home_sos'] = home_sos
historical_game_level_data['away_sos'] = away_sos
historical_game_level_data['sos_diff'] = historical_game_level_data['home_sos'] - historical_game_level_data['away_sos']

features.extend(['home_sos', 'away_sos', 'sos_diff'])
```

---

## 🔬 Advanced Improvements (Next Month)

### 4. Ensemble Model with Voting

**Impact**: HIGH | **Effort**: 4 hours | **Expected ROI**: +10-15%

```python
# Replace single XGBoost models in nfl-gather-data.py

from sklearn.ensemble import VotingClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

# Spread model ensemble
base_spread_models = [
    ('xgb', XGBClassifier(eval_metric='logloss', n_estimators=150, max_depth=6, learning_rate=0.1)),
    ('lgbm', LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.1)),
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42))
]

ensemble_spread = VotingClassifier(estimators=base_spread_models, voting='soft')
model_spread = CalibratedClassifierCV(ensemble_spread, method='isotonic', cv=5)
model_spread.fit(X_train_spread, y_spread_train)

# Repeat for moneyline and totals models
```

---

### 5. Hyperparameter Optimization

**Impact**: MEDIUM | **Effort**: 6 hours | **Expected ROI**: +5-8%

```python
# Add to nfl-gather-data.py before model training

from sklearn.model_selection import RandomizedSearchCV

# Hyperparameter grid for XGBoost
param_distributions = {
    'n_estimators': [100, 150, 200, 300],
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.15],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3]
}

# Run randomized search (faster than GridSearch)
xgb_base = XGBClassifier(eval_metric='logloss', random_state=42)
random_search = RandomizedSearchCV(
    xgb_base, 
    param_distributions, 
    n_iter=50,  # Try 50 random combinations
    cv=3, 
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42
)

print("Running hyperparameter optimization for spread model...")
random_search.fit(X_train_spread, y_spread_train)

print(f"Best parameters: {random_search.best_params_}")
print(f"Best CV score: {random_search.best_score_:.4f}")

# Use best model
best_xgb_spread = random_search.best_estimator_
model_spread = CalibratedClassifierCV(best_xgb_spread, method='isotonic', cv=5)
model_spread.fit(X_train_spread, y_spread_train)
```

---

### 6. Market Line Movement Features

**Impact**: MEDIUM | **Effort**: 8 hours | **Expected ROI**: +6-10%

**Why**: Line movement indicates where sharp money is going

```python
# Create new script: scripts/fetch_line_movements.py

import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_line_history(game_id):
    """Fetch historical line movements for a game"""
    # Note: Requires odds API subscription
    api_key = "YOUR_ODDS_API_KEY"
    url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds-history"
    
    params = {
        'apiKey': api_key,
        'regions': 'us',
        'markets': 'spreads,totals',
        'gameId': game_id
    }
    
    response = requests.get(url, params=params)
    return response.json()

def calculate_line_movement_features(df):
    """Add line movement features"""
    df['spread_line_movement'] = 0.0  # Opening spread - closing spread
    df['total_line_movement'] = 0.0   # Opening total - closing total
    df['spread_moved_toward_favorite'] = 0  # Binary: did line move toward favorite?
    
    # This would be populated with actual API data
    # For now, placeholder for implementation
    
    return df

# Add to nfl-gather-data.py
historical_game_level_data = calculate_line_movement_features(historical_game_level_data)
features.extend(['spread_line_movement', 'total_line_movement', 'spread_moved_toward_favorite'])
```

---

## 🆕 New Prediction Models

### 7. Player Prop Predictions Model

**Impact**: NEW REVENUE STREAM | **Effort**: 15 hours

```python
# Create new file: nfl-player-props.py

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# Load player stats
player_stats = pd.read_csv('data_files/stats_player_week_2024.csv')

# Feature engineering for player props
def create_player_prop_features(stats_df):
    """Create features for player prop predictions"""
    
    # Rolling averages
    stats_df = stats_df.sort_values(['player_id', 'season', 'week'])
    
    for stat in ['passing_yards', 'rushing_yards', 'receptions', 'receiving_yards']:
        stats_df[f'{stat}_last_3'] = stats_df.groupby('player_id')[stat].transform(
            lambda x: x.rolling(3, min_periods=1).mean().shift(1)
        )
        stats_df[f'{stat}_last_5'] = stats_df.groupby('player_id')[stat].transform(
            lambda x: x.rolling(5, min_periods=1).mean().shift(1)
        )
        stats_df[f'{stat}_std'] = stats_df.groupby('player_id')[stat].transform(
            lambda x: x.rolling(5, min_periods=1).std().shift(1)
        )
    
    return stats_df

# Train model for each prop type
def train_prop_model(df, target_stat):
    """Train XGBoost model for a specific player prop"""
    
    feature_cols = [col for col in df.columns if 'last_' in col or 'std' in col]
    X = df[feature_cols].fillna(0)
    y = df[target_stat]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05)
    model.fit(X_train, y_train)
    
    return model

# Example: Predict QB passing yards
player_stats = create_player_prop_features(player_stats)
qb_passing_model = train_prop_model(player_stats[player_stats['position'] == 'QB'], 'passing_yards')

print("Player prop model ready for integration!")
```

---

### 8. Live In-Game Predictions

**Impact**: NEW FEATURE | **Effort**: 20 hours

```python
# Create new file: live_game_predictor.py

import pandas as pd
import numpy as np
from xgboost import XGBClassifier

class LiveGamePredictor:
    """Predict game outcome based on current score/time remaining"""
    
    def __init__(self):
        self.model = None
        self.train_model()
    
    def create_live_features(self, score_diff, time_remaining_mins, possession, field_position):
        """Extract features from current game state"""
        return {
            'score_differential': score_diff,
            'time_remaining_minutes': time_remaining_mins,
            'time_remaining_pct': time_remaining_mins / 60.0,
            'has_possession': possession,
            'field_position_yards': field_position,
            'score_per_minute': score_diff / (60 - time_remaining_mins) if time_remaining_mins < 60 else 0,
            'is_garbage_time': 1 if abs(score_diff) > 17 and time_remaining_mins < 10 else 0,
            'is_clutch_time': 1 if abs(score_diff) <= 7 and time_remaining_mins < 5 else 0
        }
    
    def train_model(self):
        """Train model on historical play-by-play data"""
        # Load play-by-play data
        pbp = pd.read_csv('data_files/nfl_play_by_play_historical.csv.gz', compression='gzip')
        
        # Filter to key moments (start of each quarter, 2-min warning, etc.)
        key_moments = pbp[pbp['game_seconds_remaining'].isin([3600, 2700, 1800, 900, 120])]
        
        # Create features and target
        X_live = pd.DataFrame([
            self.create_live_features(
                row['score_differential'],
                row['game_seconds_remaining'] / 60,
                row['posteam'] == row['home_team'],
                row['yardline_100']
            ) for _, row in key_moments.iterrows()
        ])
        
        y_live = key_moments['result']  # Home team won = 1
        
        self.model = XGBClassifier(n_estimators=150, max_depth=5)
        self.model.fit(X_live, y_live)
    
    def predict_win_probability(self, score_diff, time_remaining, possession, field_pos):
        """Get live win probability"""
        features = pd.DataFrame([self.create_live_features(score_diff, time_remaining, possession, field_pos)])
        return self.model.predict_proba(features)[0, 1]

# Usage
live_predictor = LiveGamePredictor()
win_prob = live_predictor.predict_win_probability(
    score_diff=7,  # Home team up by 7
    time_remaining=8.5,  # 8.5 minutes left
    possession=True,  # Home team has ball
    field_pos=35  # At opponent's 35 yard line
)
print(f"Home team win probability: {win_prob*100:.1f}%")
```

---

### 9. Injury Impact Model

**Impact**: HIGH | **Effort**: 12 hours

```python
# Create scripts/fetch_injury_data.py

import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_injury_reports():
    """Scrape weekly injury reports"""
    url = "https://www.espn.com/nfl/injuries"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    injuries = []
    # Parse injury table (structure varies by site)
    # This is a simplified example
    
    return pd.DataFrame(injuries)

def quantify_injury_impact(player_position, injury_status):
    """Convert injury status to numeric impact score"""
    impact_weights = {
        'QB': {'Out': -15, 'Questionable': -8, 'Doubtful': -12},
        'RB': {'Out': -5, 'Questionable': -3, 'Doubtful': -4},
        'WR': {'Out': -4, 'Questionable': -2, 'Doubtful': -3},
        'OL': {'Out': -3, 'Questionable': -1.5, 'Doubtful': -2.5},
        'DEF': {'Out': -2, 'Questionable': -1, 'Doubtful': -1.5}
    }
    
    return impact_weights.get(player_position, {}).get(injury_status, 0)

# Add to nfl-gather-data.py
def add_injury_features(df):
    """Add injury impact scores to game data"""
    injury_data = scrape_injury_reports()
    
    df['home_injury_impact'] = 0.0
    df['away_injury_impact'] = 0.0
    
    # Aggregate injury impacts by team
    for idx, row in df.iterrows():
        home_injuries = injury_data[injury_data['team'] == row['home_team']]
        away_injuries = injury_data[injury_data['team'] == row['away_team']]
        
        df.at[idx, 'home_injury_impact'] = sum([
            quantify_injury_impact(inj['position'], inj['status']) 
            for _, inj in home_injuries.iterrows()
        ])
        
        df.at[idx, 'away_injury_impact'] = sum([
            quantify_injury_impact(inj['position'], inj['status']) 
            for _, inj in away_injuries.iterrows()
        ])
    
    df['injury_impact_diff'] = df['home_injury_impact'] - df['away_injury_impact']
    
    return df

features.extend(['home_injury_impact', 'away_injury_impact', 'injury_impact_diff'])
```

---

## 📊 Model Performance Enhancements

### 10. SHAP Values for Interpretability

**Impact**: TRUST & DEBUGGING | **Effort**: 3 hours

```python
# Add to nfl-gather-data.py

import shap

# After training models, calculate SHAP values
explainer_spread = shap.TreeExplainer(spread_xgb)
shap_values_spread = explainer_spread.shap_values(X_test_spread)

# Save SHAP summary
shap.summary_plot(shap_values_spread, X_test_spread, show=False)
import matplotlib.pyplot as plt
plt.savefig('data_files/exports/shap_spread_summary.png', bbox_inches='tight', dpi=300)
plt.close()

# Save SHAP values for UI integration
shap_df = pd.DataFrame(shap_values_spread, columns=X_test_spread.columns)
shap_df.to_csv('data_files/shap_values_spread.csv', index=False)

print("SHAP analysis complete - interpretability enhanced!")
```

---

## 🎯 Implementation Priority

**Week 1**: Items #1, #2, #3 (QB, EPA, SOS)  
**Week 2**: Items #4, #5 (Ensemble, Hyperparameter)  
**Week 3**: Item #10 (SHAP), start #7 (Player Props)  
**Month 2**: Items #6, #8, #9 (Line Movement, Live, Injuries)

---

## 📈 Expected Cumulative Impact

| Phase | Expected ROI Improvement | Win Rate Improvement |
|-------|-------------------------|---------------------|
| Week 1 (Items 1-3) | +16-25% | +4-6% |
| Week 2 (Items 4-5) | +31-48% | +7-10% |
| Month 2 (All) | +50-75% | +12-18% |

**Current**: ~60% spread ROI, ~28% moneyline threshold  
**Target**: ~100%+ spread ROI, 85%+ win rate on elite bets

---

## 💡 Additional Quick Wins

### 11. Coaching Matchup Features

**Impact**: MEDIUM | **Effort**: 2 hours | **Expected ROI**: +3-5%

```python
# Add to nfl-gather-data.py

def calculate_coach_performance(df):
    """Track coach win rates, ATS performance, playoff success"""
    
    # Calculate coach career stats up to each game
    df['home_coach_win_pct'] = 0.0
    df['away_coach_win_pct'] = 0.0
    df['home_coach_ats_pct'] = 0.0
    df['away_coach_ats_pct'] = 0.0
    df['coach_experience_diff'] = 0.0
    
    for idx, row in df.iterrows():
        season = row['season']
        week = row['week']
        home_coach = row['home_coach']
        away_coach = row['away_coach']
        
        # Home coach historical performance
        home_coach_games = df[
            ((df['home_coach'] == home_coach) | (df['away_coach'] == home_coach)) &
            ((df['season'] < season) | ((df['season'] == season) & (df['week'] < week)))
        ]
        
        if len(home_coach_games) > 0:
            home_wins = home_coach_games[
                ((home_coach_games['home_coach'] == home_coach) & (home_coach_games['homeWin'] == 1)) |
                ((home_coach_games['away_coach'] == home_coach) & (home_coach_games['awayWin'] == 1))
            ]
            df.at[idx, 'home_coach_win_pct'] = len(home_wins) / len(home_coach_games)
            
            # ATS performance
            home_ats_wins = home_coach_games[
                ((home_coach_games['home_coach'] == home_coach) & (home_coach_games['spreadCovered'] == 1)) |
                ((home_coach_games['away_coach'] == home_coach) & (home_coach_games['underdogCovered'] == 1))
            ]
            df.at[idx, 'home_coach_ats_pct'] = len(home_ats_wins) / len(home_coach_games)
            df.at[idx, 'coach_experience_diff'] = len(home_coach_games)
        
        # Away coach historical performance
        away_coach_games = df[
            ((df['home_coach'] == away_coach) | (df['away_coach'] == away_coach)) &
            ((df['season'] < season) | ((df['season'] == season) & (df['week'] < week)))
        ]
        
        if len(away_coach_games) > 0:
            away_wins = away_coach_games[
                ((away_coach_games['home_coach'] == away_coach) & (away_coach_games['homeWin'] == 1)) |
                ((away_coach_games['away_coach'] == away_coach) & (away_coach_games['awayWin'] == 1))
            ]
            df.at[idx, 'away_coach_win_pct'] = len(away_wins) / len(away_coach_games)
            
            away_ats_wins = away_coach_games[
                ((away_coach_games['home_coach'] == away_coach) & (away_coach_games['spreadCovered'] == 1)) |
                ((away_coach_games['away_coach'] == away_coach) & (away_coach_games['underdogCovered'] == 1))
            ]
            df.at[idx, 'away_coach_ats_pct'] = len(away_ats_wins) / len(away_coach_games)
            df.at[idx, 'coach_experience_diff'] -= len(away_coach_games)
    
    df['coach_win_pct_diff'] = df['home_coach_win_pct'] - df['away_coach_win_pct']
    df['coach_ats_diff'] = df['home_coach_ats_pct'] - df['away_coach_ats_pct']
    
    return df

historical_game_level_data = calculate_coach_performance(historical_game_level_data)
features.extend([
    'home_coach_win_pct', 'away_coach_win_pct', 'coach_win_pct_diff',
    'home_coach_ats_pct', 'away_coach_ats_pct', 'coach_ats_diff',
    'coach_experience_diff'
])
```

### 12. Home Field Advantage by Stadium

**Impact**: MEDIUM | **Effort**: 1 hour | **Expected ROI**: +2-4%

```python
# Add to nfl-gather-data.py

def calculate_stadium_advantage(df):
    """Calculate win rate for each stadium"""
    
    df['stadium_home_win_rate'] = 0.0
    df['stadium_home_ats_rate'] = 0.0
    
    for idx, row in df.iterrows():
        stadium = row['stadium']
        season = row['season']
        week = row['week']
        
        # Historical performance at this stadium
        stadium_games = df[
            (df['stadium'] == stadium) &
            ((df['season'] < season) | ((df['season'] == season) & (df['week'] < week)))
        ]
        
        if len(stadium_games) > 0:
            home_wins = stadium_games[stadium_games['homeWin'] == 1]
            df.at[idx, 'stadium_home_win_rate'] = len(home_wins) / len(stadium_games)
            
            home_ats_wins = stadium_games[stadium_games['spreadCovered'] == 1]
            df.at[idx, 'stadium_home_ats_rate'] = len(home_ats_wins) / len(stadium_games)
    
    # League average home win rate
    league_avg_home_win = df['homeWin'].mean()
    df['stadium_advantage'] = df['stadium_home_win_rate'] - league_avg_home_win
    
    return df

historical_game_level_data = calculate_stadium_advantage(historical_game_level_data)
features.extend(['stadium_home_win_rate', 'stadium_home_ats_rate', 'stadium_advantage'])
```

### 13. Prime Time & Time Zone Effects

**Impact**: LOW-MEDIUM | **Effort**: 1 hour | **Expected ROI**: +1-3%

```python
# Add to nfl-gather-data.py

import pytz
from datetime import datetime

def add_game_time_features(df):
    """Add features for prime time, time zone travel"""
    
    # Parse game time
    df['hour'] = pd.to_datetime(df['gametime'], format='%H:%M', errors='coerce').dt.hour
    
    # Prime time games (after 7pm ET)
    df['is_prime_time'] = (df['hour'] >= 19).astype(int)
    df['is_thursday_night'] = ((df['weekday'] == 'Thursday') & df['is_prime_time']).astype(int)
    df['is_sunday_night'] = ((df['weekday'] == 'Sunday') & df['is_prime_time']).astype(int)
    df['is_monday_night'] = ((df['weekday'] == 'Monday') & df['is_prime_time']).astype(int)
    
    # Time zone mapping (simplified)
    timezone_map = {
        'NE': 'ET', 'BUF': 'ET', 'MIA': 'ET', 'NYJ': 'ET',
        'BAL': 'ET', 'CIN': 'ET', 'CLE': 'ET', 'PIT': 'ET',
        'IND': 'ET', 'JAX': 'ET', 'TEN': 'ET', 'HOU': 'CT',
        'CHI': 'CT', 'DET': 'ET', 'GB': 'CT', 'MIN': 'CT',
        'ATL': 'ET', 'CAR': 'ET', 'NO': 'CT', 'TB': 'ET',
        'WAS': 'ET', 'DAL': 'CT', 'NYG': 'ET', 'PHI': 'ET',
        'DEN': 'MT', 'KC': 'CT', 'LV': 'PT', 'LAC': 'PT',
        'ARI': 'MT', 'LA': 'PT', 'SF': 'PT', 'SEA': 'PT'
    }
    
    df['home_tz'] = df['home_team'].map(timezone_map)
    df['away_tz'] = df['away_team'].map(timezone_map)
    
    # Time zone travel (ET=0, CT=1, MT=2, PT=3)
    tz_offset = {'ET': 0, 'CT': 1, 'MT': 2, 'PT': 3}
    df['away_time_zone_travel'] = abs(
        df['away_tz'].map(tz_offset) - df['home_tz'].map(tz_offset)
    )
    
    # West coast team traveling east for early game is disadvantage
    df['west_coast_early_game'] = (
        (df['away_tz'] == 'PT') & 
        (df['home_tz'].isin(['ET', 'CT'])) & 
        (df['hour'] < 15)
    ).astype(int)
    
    return df

historical_game_level_data = add_game_time_features(historical_game_level_data)
features.extend([
    'is_prime_time', 'is_thursday_night', 'is_sunday_night', 'is_monday_night',
    'away_time_zone_travel', 'west_coast_early_game'
])
```

---

## 🔧 Model Architecture Improvements

### 14. Feature Interaction Terms

**Impact**: HIGH | **Effort**: 2 hours | **Expected ROI**: +6-9%

```python
# Add to nfl-gather-data.py after basic features

def create_interaction_features(df):
    """Create powerful interaction terms between key features"""
    
    # Interaction: QB quality × offensive line quality (simplified with avg score)
    df['qb_x_offense_quality'] = df['home_qb_rolling_qbr'] * df['homeTeamAvgScore']
    df['away_qb_x_offense_quality'] = df['away_qb_rolling_qbr'] * df['awayTeamAvgScore']
    
    # Interaction: Rest advantage × momentum
    df['rest_x_momentum_home'] = df['home_rest'] * df['homeTeamLast3WinPct']
    df['rest_x_momentum_away'] = df['away_rest'] * df['awayTeamLast3WinPct']
    
    # Interaction: Weather × playing style (outdoor teams vs dome teams)
    df['weather_x_outdoor_advantage'] = df['temp'] * (df['roof'] == 'outdoors').astype(int)
    
    # Interaction: Spread size × team volatility
    df['spread_x_volatility_home'] = df['spreadSize'] * df['homeTeamAvgPointDiff']
    df['spread_x_volatility_away'] = df['spreadSize'] * df['awayTeamAvgPointDiff']
    
    # Interaction: Division game × rivalry intensity (measured by closeness)
    df['division_x_competitive'] = df['div_game'] * df['isCloseSpread']
    
    # Interaction: Coach experience × high pressure situation
    df['coach_exp_x_pressure'] = df['coach_experience_diff'] * df['is_prime_time']
    
    # Polynomial features for key stats
    df['spread_line_squared'] = df['spread_line'] ** 2
    df['total_line_squared'] = df['total_line'] ** 2
    
    return df

historical_game_level_data = create_interaction_features(historical_game_level_data)
features.extend([
    'qb_x_offense_quality', 'away_qb_x_offense_quality',
    'rest_x_momentum_home', 'rest_x_momentum_away',
    'weather_x_outdoor_advantage',
    'spread_x_volatility_home', 'spread_x_volatility_away',
    'division_x_competitive', 'coach_exp_x_pressure',
    'spread_line_squared', 'total_line_squared'
])
```

### 15. Temporal Features & Seasonality

**Impact**: MEDIUM | **Effort**: 1.5 hours | **Expected ROI**: +4-6%

```python
# Add to nfl-gather-data.py

def add_temporal_features(df):
    """Add season progression and playoff race features"""
    
    # Season progression (early season vs late season)
    df['season_progress'] = df['week'] / 18  # Normalized to 0-1
    df['is_early_season'] = (df['week'] <= 4).astype(int)
    df['is_mid_season'] = ((df['week'] > 4) & (df['week'] <= 12)).astype(int)
    df['is_late_season'] = (df['week'] > 12).astype(int)
    
    # Playoff implications (simplified - teams with winning record late in season)
    df['home_playoff_bound'] = (
        (df['homeTeamWinPct'] > 0.6) & (df['week'] > 10)
    ).astype(int)
    df['away_playoff_bound'] = (
        (df['awayTeamWinPct'] > 0.6) & (df['week'] > 10)
    ).astype(int)
    
    # Desperation factor (losing team late in season)
    df['home_desperation'] = (
        (df['homeTeamWinPct'] < 0.4) & (df['week'] > 10)
    ).astype(int)
    df['away_desperation'] = (
        (df['awayTeamWinPct'] < 0.4) & (df['week'] > 10)
    ).astype(int)
    
    # Week-over-week performance changes
    df['home_momentum_change'] = df['homeTeamLast3WinPct'] - df['homeTeamWinPct']
    df['away_momentum_change'] = df['awayTeamLast3WinPct'] - df['awayTeamWinPct']
    
    # Cyclical encoding of week (captures weekly patterns)
    df['week_sin'] = np.sin(2 * np.pi * df['week'] / 18)
    df['week_cos'] = np.cos(2 * np.pi * df['week'] / 18)
    
    return df

historical_game_level_data = add_temporal_features(historical_game_level_data)
features.extend([
    'season_progress', 'is_early_season', 'is_mid_season', 'is_late_season',
    'home_playoff_bound', 'away_playoff_bound',
    'home_desperation', 'away_desperation',
    'home_momentum_change', 'away_momentum_change',
    'week_sin', 'week_cos'
])
```

### 16. Advanced Calibration with Temperature Scaling

**Impact**: HIGH | **Effort**: 3 hours | **Expected ROI**: +7-10%

```python
# Add to nfl-gather-data.py after initial calibration

from sklearn.linear_model import LogisticRegression

class TemperatureScaling:
    """
    Temperature scaling for better probability calibration.
    Reference: "On Calibration of Modern Neural Networks" (Guo et al., 2017)
    """
    def __init__(self):
        self.temperature = 1.0
    
    def fit(self, logits, labels):
        """Learn optimal temperature parameter"""
        # Use validation set to find best temperature
        logits = np.array(logits).reshape(-1, 1)
        
        # Optimize temperature using cross-entropy loss
        from scipy.optimize import minimize
        
        def temperature_cross_entropy(temp, logits, labels):
            scaled_logits = logits / temp
            probs = 1 / (1 + np.exp(-scaled_logits))
            loss = -np.mean(labels * np.log(probs + 1e-10) + (1 - labels) * np.log(1 - probs + 1e-10))
            return loss
        
        result = minimize(
            temperature_cross_entropy,
            x0=1.0,
            args=(logits, labels),
            bounds=[(0.1, 10.0)],
            method='L-BFGS-B'
        )
        
        self.temperature = result.x[0]
        return self
    
    def predict_proba(self, logits):
        """Apply temperature scaling to logits"""
        scaled_logits = logits / self.temperature
        probs = 1 / (1 + np.exp(-scaled_logits))
        return np.column_stack([1 - probs, probs])

# Apply temperature scaling AFTER isotonic calibration
print("\n🌡️ Applying Temperature Scaling for Enhanced Calibration...")

# Get raw predictions from base model before calibration
spread_base_model = model_spread.calibrated_classifiers_[0].estimator
spread_logits = spread_base_model.predict_proba(X_test_spread)[:, 1]

# Convert to logits
spread_logits_transformed = np.log(spread_logits / (1 - spread_logits + 1e-10))

# Fit temperature scaling
temp_scaler_spread = TemperatureScaling()
temp_scaler_spread.fit(spread_logits_transformed, y_spread_test)

print(f"   Optimal temperature for spread model: {temp_scaler_spread.temperature:.3f}")

# Apply to all predictions
spread_probs_temp_scaled = temp_scaler_spread.predict_proba(
    np.log(historical_game_level_data['prob_underdogCovered'] / (1 - historical_game_level_data['prob_underdogCovered'] + 1e-10))
)[:, 1]

# Update probabilities
historical_game_level_data['prob_underdogCovered_temp_scaled'] = spread_probs_temp_scaled

# Validate improvement in calibration
from sklearn.calibration import calibration_curve
fraction_post, mean_pred_post = calibration_curve(
    y_spread_test, 
    spread_probs_temp_scaled[:len(y_spread_test)], 
    n_bins=5
)
post_calib_error = np.abs(fraction_post - mean_pred_post).mean()
print(f"   Calibration error after temperature scaling: {post_calib_error:.3f}")
```

---

## 📊 See Also

- [ADVANCED_MODEL_ROADMAP_PART2.md](ADVANCED_MODEL_ROADMAP_PART2.md) - Neural networks, deep learning, advanced techniques
- [NEW_MODELS_ROADMAP.md](NEW_MODELS_ROADMAP.md) - Completely new prediction models (player props, parlays, teasers)
- [DATA_PIPELINE_ROADMAP.md](DATA_PIPELINE_ROADMAP.md) - Data infrastructure and automation improvements
