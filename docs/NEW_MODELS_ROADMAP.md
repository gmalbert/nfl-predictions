# 🎲 New Prediction Models Roadmap

**Completely New Model Types & Revenue Streams**  
**Focus**: Player props, parlays, live betting, exotic markets

---

## 🏈 Player Performance Models

### 24. Quarterback Passing Yards Model

**Impact**: NEW MARKET | **Effort**: 10 hours | **Revenue**: High volume prop market

```python
# Create new file: models/qb_passing_yards.py

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

class QBPassingYardsPredictor:
    """Predict QB passing yards for player props betting"""
    
    def __init__(self):
        self.model = None
        self.feature_columns = None
        
    def create_qb_features(self, qb_stats, opponent_def_stats):
        """Create comprehensive QB + matchup features"""
        
        features = pd.DataFrame()
        
        # QB historical averages
        features['qb_avg_passing_yards_last_3'] = qb_stats.groupby('player_id')['passing_yards'].transform(
            lambda x: x.rolling(3, min_periods=1).mean().shift(1)
        )
        features['qb_avg_passing_yards_last_5'] = qb_stats.groupby('player_id')['passing_yards'].transform(
            lambda x: x.rolling(5, min_periods=1).mean().shift(1)
        )
        features['qb_avg_passing_yards_season'] = qb_stats.groupby(['player_id', 'season'])['passing_yards'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        
        # QB volatility
        features['qb_passing_yards_std'] = qb_stats.groupby('player_id')['passing_yards'].transform(
            lambda x: x.rolling(5, min_periods=1).std().shift(1)
        )
        
        # QB completion percentage
        features['qb_completion_pct_last_3'] = (
            qb_stats.groupby('player_id')['completions'].transform(lambda x: x.rolling(3, min_periods=1).mean().shift(1)) /
            qb_stats.groupby('player_id')['attempts'].transform(lambda x: x.rolling(3, min_periods=1).mean().shift(1))
        )
        
        # QB touchdown rate
        features['qb_td_rate_last_5'] = (
            qb_stats.groupby('player_id')['passing_tds'].transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
        )
        
        # QB yards per attempt
        features['qb_ypa_last_3'] = (
            qb_stats.groupby('player_id')['passing_yards'].transform(lambda x: x.rolling(3, min_periods=1).mean().shift(1)) /
            qb_stats.groupby('player_id')['attempts'].transform(lambda x: x.rolling(3, min_periods=1).mean().shift(1))
        )
        
        # Opponent defense stats
        features['opp_def_passing_yards_allowed_avg'] = opponent_def_stats.groupby('team')['passing_yards_allowed'].transform(
            lambda x: x.rolling(3, min_periods=1).mean().shift(1)
        )
        features['opp_def_sacks_per_game'] = opponent_def_stats.groupby('team')['sacks'].transform(
            lambda x: x.rolling(3, min_periods=1).mean().shift(1)
        )
        features['opp_def_interceptions_per_game'] = opponent_def_stats.groupby('team')['interceptions'].transform(
            lambda x: x.rolling(3, min_periods=1).mean().shift(1)
        )
        
        # Matchup factors
        features['is_home_game'] = qb_stats['is_home'].astype(int)
        features['is_division_game'] = qb_stats['is_division'].astype(int)
        features['is_prime_time'] = qb_stats['is_prime_time'].astype(int)
        
        # Weather
        features['temp'] = qb_stats['temp']
        features['wind'] = qb_stats['wind']
        features['is_dome'] = (qb_stats['roof'] == 'dome').astype(int)
        
        # Game script indicators (team strength differential)
        features['team_win_pct'] = qb_stats['team_win_pct']
        features['opp_win_pct'] = qb_stats['opp_win_pct']
        features['win_pct_diff'] = features['team_win_pct'] - features['opp_win_pct']
        
        # Vegas totals (game total suggests pace/scoring)
        features['game_total'] = qb_stats['total_line']
        
        return features
    
    def train(self, qb_stats, opponent_stats):
        """Train QB passing yards prediction model"""
        
        # Create features
        X = self.create_qb_features(qb_stats, opponent_stats)
        y = qb_stats['passing_yards']
        
        # Remove NaN rows
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_idx]
        y = y[valid_idx]
        
        self.feature_columns = X.columns.tolist()
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.model = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"\n📊 QB Passing Yards Model Performance:")
        print(f"   MAE: {mae:.2f} yards")
        print(f"   RMSE: {rmse:.2f} yards")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔝 Top 5 Features:")
        print(feature_importance.head().to_string(index=False))
        
        return self.model
    
    def predict(self, qb_features):
        """Predict passing yards for a QB in upcoming game"""
        prediction = self.model.predict(qb_features[self.feature_columns])
        
        # Also get prediction interval using quantile regression
        lower_bound = prediction * 0.75  # Simplified; use quantile regression for better bounds
        upper_bound = prediction * 1.25
        
        return {
            'predicted_yards': prediction[0],
            'lower_bound': lower_bound[0],
            'upper_bound': upper_bound[0]
        }

# Usage
qb_model = QBPassingYardsPredictor()
qb_model.train(qb_historical_stats, defensive_stats)

# Predict for upcoming game
prediction = qb_model.predict(mahomes_week_17_features)
print(f"Patrick Mahomes projected: {prediction['predicted_yards']:.1f} yards")
print(f"Range: {prediction['lower_bound']:.1f} - {prediction['upper_bound']:.1f}")
```

---

### 25. Running Back Yards + TD Model

**Impact**: NEW MARKET | **Effort**: 8 hours

```python
# Create new file: models/rb_performance.py

class RBPerformancePredictor:
    """Predict RB rushing yards and TDs"""
    
    def __init__(self):
        self.yards_model = None
        self.td_model = None
        
    def create_rb_features(self, rb_stats, opponent_def):
        """Create RB-specific features"""
        
        features = pd.DataFrame()
        
        # RB volume metrics
        features['rb_carries_last_3'] = rb_stats.groupby('player_id')['carries'].transform(
            lambda x: x.rolling(3, min_periods=1).mean().shift(1)
        )
        features['rb_targets_last_3'] = rb_stats.groupby('player_id')['targets'].transform(
            lambda x: x.rolling(3, min_periods=1).mean().shift(1)
        )
        features['rb_touch_share'] = rb_stats['touches'] / rb_stats['team_total_touches']
        
        # Efficiency metrics
        features['rb_yards_per_carry_last_3'] = (
            rb_stats.groupby('player_id')['rushing_yards'].transform(lambda x: x.rolling(3).mean().shift(1)) /
            rb_stats.groupby('player_id')['carries'].transform(lambda x: x.rolling(3).mean().shift(1))
        )
        
        features['rb_yards_after_contact'] = rb_stats['yards_after_contact']
        features['rb_broken_tackles_per_touch'] = rb_stats['broken_tackles'] / rb_stats['touches']
        
        # Opponent run defense
        features['opp_rush_yards_allowed_per_game'] = opponent_def['rush_yards_allowed_avg']
        features['opp_rush_tds_allowed_per_game'] = opponent_def['rush_tds_allowed_avg']
        features['opp_def_run_stop_rate'] = opponent_def['run_stop_rate']
        
        # Game script
        features['team_spread'] = rb_stats['spread_line']  # Negative = favored (more likely to run)
        features['game_total'] = rb_stats['total_line']
        
        # Red zone usage
        features['rb_red_zone_carry_share'] = rb_stats['red_zone_carries'] / rb_stats['team_red_zone_carries']
        features['rb_goal_line_carry_share'] = rb_stats['goal_line_carries'] / rb_stats['team_goal_line_carries']
        
        return features
    
    def train(self, rb_stats, defense_stats):
        """Train yards and TD models separately"""
        
        X = self.create_rb_features(rb_stats, defense_stats)
        
        # Yards model (regression)
        y_yards = rb_stats['rushing_yards']
        valid_yards = ~(X.isna().any(axis=1) | y_yards.isna())
        
        self.yards_model = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05)
        self.yards_model.fit(X[valid_yards], y_yards[valid_yards])
        
        # TD model (classification for 0, 1, 2+ TDs)
        y_tds = rb_stats['rushing_tds'].clip(upper=2)  # 0, 1, or 2+
        valid_tds = ~(X.isna().any(axis=1) | y_tds.isna())
        
        from xgboost import XGBClassifier
        self.td_model = XGBClassifier(n_estimators=200, max_depth=4)
        self.td_model.fit(X[valid_tds], y_tds[valid_tds])
        
        return self
    
    def predict(self, rb_features):
        """Predict yards and TD probability"""
        yards_pred = self.yards_model.predict(rb_features)[0]
        td_probs = self.td_model.predict_proba(rb_features)[0]
        
        return {
            'projected_yards': yards_pred,
            'prob_0_tds': td_probs[0],
            'prob_1_td': td_probs[1] if len(td_probs) > 1 else 0,
            'prob_2plus_tds': td_probs[2] if len(td_probs) > 2 else 0,
            'expected_tds': sum(i * prob for i, prob in enumerate(td_probs))
        }

# Usage
rb_model = RBPerformancePredictor()
rb_model.train(rb_historical_stats, defense_stats)

prediction = rb_model.predict(henry_week_17_features)
print(f"Derrick Henry: {prediction['projected_yards']:.1f} yards, {prediction['expected_tds']:.2f} TDs")
```

---

### 26. Wide Receiver Receptions + Yards Model

**Impact**: NEW MARKET | **Effort**: 8 hours

```python
# Create new file: models/wr_performance.py

class WRPerformancePredictor:
    """Predict WR receptions and receiving yards"""
    
    def create_wr_features(self, wr_stats, qb_stats, defense_stats):
        """WR features including QB chemistry"""
        
        features = pd.DataFrame()
        
        # WR usage
        features['wr_target_share'] = wr_stats['targets'] / wr_stats['team_total_targets']
        features['wr_targets_last_3'] = wr_stats.groupby('player_id')['targets'].transform(
            lambda x: x.rolling(3).mean().shift(1)
        )
        features['wr_air_yards_share'] = wr_stats['air_yards'] / wr_stats['team_total_air_yards']
        
        # Efficiency
        features['wr_catch_rate'] = wr_stats['receptions'] / wr_stats['targets']
        features['wr_yards_per_route_run'] = wr_stats['receiving_yards'] / wr_stats['routes_run']
        features['wr_yac_per_reception'] = wr_stats['yards_after_catch'] / wr_stats['receptions']
        
        # QB quality effect
        features['qb_completion_pct'] = qb_stats['completion_pct']
        features['qb_passing_yards_per_game'] = qb_stats['passing_yards_avg']
        features['qb_target_distribution'] = 1 / qb_stats['num_receivers_targeted']  # QB focuses targets
        
        # Opponent secondary
        features['opp_pass_def_yards_allowed'] = defense_stats['pass_yards_allowed_avg']
        features['opp_slot_coverage_rating'] = defense_stats['slot_coverage_grade']
        features['opp_cb_quality'] = defense_stats['cb_rating']
        
        # Matchup specifics
        features['wr_vs_cb_matchup_rating'] = wr_stats['matchup_advantage_score']
        features['is_slot_receiver'] = wr_stats['slot_rate']
        
        # Game environment
        features['game_pace'] = wr_stats['expected_plays']
        features['team_pass_rate'] = wr_stats['team_pass_rate']
        
        return features
    
    # Similar structure to RB model with separate reception and yards predictions
```

---

## 🎰 Parlay & Teaser Models

### 27. Optimal Parlay Builder

**Impact**: HIGH EDGE | **Effort**: 12 hours

```python
# Create new file: models/parlay_optimizer.py

from itertools import combinations
import numpy as np

class ParlayOptimizer:
    """Find +EV parlays using correlation modeling"""
    
    def __init__(self, predictions_df):
        self.predictions = predictions_df
        self.correlation_matrix = None
        
    def calculate_game_correlations(self):
        """Calculate correlations between game outcomes"""
        
        # Correlations to model:
        # 1. Same-game correlations (spread + total)
        # 2. Division game correlations (rivals playing simultaneously)
        # 3. Weather correlations (multiple games in bad weather)
        # 4. Conference correlations (AFC/NFC strength)
        
        corr_matrix = pd.DataFrame()
        
        for i, game1 in self.predictions.iterrows():
            for j, game2 in self.predictions.iterrows():
                if i >= j:
                    continue
                
                # Calculate correlation score
                corr_score = 0.0
                
                # Same game spread/total correlation
                if game1['game_id'] == game2['game_id']:
                    corr_score += 0.3  # Moderate positive correlation
                
                # Shared team (impossible but check for data issues)
                shared_teams = set([game1['home_team'], game1['away_team']]) & \
                               set([game2['home_team'], game2['away_team']])
                if shared_teams:
                    corr_score = 1.0  # Perfect correlation (same outcome)
                
                # Division rivals playing simultaneously
                if (game1['div_game'] and game2['div_game'] and 
                    game1['gameday'] == game2['gameday']):
                    corr_score += 0.15
                
                # Weather correlation
                if (game1['stadium'] == game2['stadium'] or
                    (game1['temp'] < 32 and game2['temp'] < 32)):
                    corr_score += 0.1
                
                corr_matrix.loc[game1['game_id'], game2['game_id']] = corr_score
        
        self.correlation_matrix = corr_matrix
        return corr_matrix
    
    def calculate_parlay_true_odds(self, games, bet_types):
        """
        Calculate true odds of parlay accounting for correlation
        
        Args:
            games: list of game_ids
            bet_types: list of bet types ('spread', 'total_over', 'total_under', 'moneyline')
        """
        
        # Get individual probabilities
        probs = []
        for game_id, bet_type in zip(games, bet_types):
            game_row = self.predictions[self.predictions['game_id'] == game_id].iloc[0]
            
            if bet_type == 'spread':
                prob = game_row['prob_underdogCovered']
            elif bet_type == 'total_over':
                prob = game_row['prob_overHit']
            elif bet_type == 'total_under':
                prob = 1 - game_row['prob_overHit']
            elif bet_type == 'moneyline':
                prob = game_row['prob_underdogWon']
            
            probs.append(prob)
        
        # Naive probability (assumes independence)
        naive_prob = np.prod(probs)
        
        # Adjust for correlation
        # Use Gaussian copula for simplicity
        correlation_adjustment = 1.0
        
        for i in range(len(games)):
            for j in range(i+1, len(games)):
                if games[i] in self.correlation_matrix.index and \
                   games[j] in self.correlation_matrix.columns:
                    corr = self.correlation_matrix.loc[games[i], games[j]]
                    # Positive correlation increases joint probability
                    correlation_adjustment *= (1 + corr * 0.1)
        
        adjusted_prob = naive_prob * correlation_adjustment
        
        return adjusted_prob, naive_prob
    
    def find_positive_ev_parlays(self, min_legs=2, max_legs=4, min_ev=0.05):
        """Find all +EV parlays"""
        
        positive_ev_parlays = []
        
        # Get all games with decent probability (>40% to avoid huge longshots)
        viable_bets = []
        
        for idx, game in self.predictions.iterrows():
            if game['prob_underdogCovered'] > 0.4:
                viable_bets.append((game['game_id'], 'spread', game['prob_underdogCovered'], -110))
            if game['prob_overHit'] > 0.4:
                viable_bets.append((game['game_id'], 'total_over', game['prob_overHit'], -110))
            if 1 - game['prob_overHit'] > 0.4:
                viable_bets.append((game['game_id'], 'total_under', 1 - game['prob_overHit'], -110))
        
        # Try all combinations
        for n_legs in range(min_legs, max_legs + 1):
            for combo in combinations(viable_bets, n_legs):
                games = [bet[0] for bet in combo]
                bet_types = [bet[1] for bet in combo]
                odds_list = [bet[3] for bet in combo]
                
                # Calculate parlay odds
                parlay_decimal_odds = 1.0
                for odds in odds_list:
                    if odds < 0:
                        parlay_decimal_odds *= (1 + 100/abs(odds))
                    else:
                        parlay_decimal_odds *= (1 + odds/100)
                
                # Get true probability
                true_prob, naive_prob = self.calculate_parlay_true_odds(games, bet_types)
                
                # Calculate EV
                ev = (true_prob * (parlay_decimal_odds - 1)) - (1 - true_prob)
                
                if ev > min_ev:
                    positive_ev_parlays.append({
                        'games': games,
                        'bet_types': bet_types,
                        'true_probability': true_prob,
                        'parlay_odds': parlay_decimal_odds,
                        'expected_value': ev,
                        'roi': ev * 100
                    })
        
        # Sort by EV
        positive_ev_parlays.sort(key=lambda x: x['expected_value'], reverse=True)
        
        return pd.DataFrame(positive_ev_parlays)

# Usage
parlay_finder = ParlayOptimizer(predictions_df)
parlay_finder.calculate_game_correlations()

best_parlays = parlay_finder.find_positive_ev_parlays(min_legs=2, max_legs=3, min_ev=0.03)
print(f"\n🎰 Found {len(best_parlays)} +EV parlays!")
print(best_parlays.head(10))
```

---

### 28. Teaser Optimizer

**Impact**: MEDIUM | **Effort**: 6 hours

```python
# Create new file: models/teaser_optimizer.py

class TeaserOptimizer:
    """Optimize 6-point and 7-point teasers"""
    
    def __init__(self, predictions_df):
        self.predictions = predictions_df
        
    def find_key_numbers(self):
        """Identify spreads near key numbers (3, 7, 10, 14)"""
        key_numbers = [3, 7, 10, 14]
        
        self.predictions['teaser_value'] = 0.0
        
        for idx, row in self.predictions.iterrows():
            spread = row['spread_line']
            value = 0
            
            # 6-point teaser analysis
            for key in key_numbers:
                # Moving through a key number is very valuable
                if abs(spread) <= key and abs(spread) + 6 > key:
                    value += 1
                if abs(spread) >= key and abs(spread) - 6 < key:
                    value += 1
            
            self.predictions.at[idx, 'teaser_value'] = value
        
        return self.predictions[self.predictions['teaser_value'] > 0].sort_values('teaser_value', ascending=False)
    
    def calculate_teaser_probability(self, spread, tease_points, is_favorite):
        """
        Calculate win probability for teased spread
        Using historical teaser data and key number crossing
        """
        
        # Adjust spread by tease points
        if is_favorite:
            adjusted_spread = spread - tease_points  # Make spread smaller (better for favorite)
        else:
            adjusted_spread = spread + tease_points  # Make spread larger (better for underdog)
        
        # Historical win rates by teased spread
        # These are empirical values from historical teaser performance
        teaser_win_rates = {
            range(-14, -10): 0.75,
            range(-10, -7): 0.80,
            range(-7, -3): 0.85,
            range(-3, 0): 0.88,
            range(0, 3): 0.88,
            range(3, 7): 0.85,
            range(7, 10): 0.80,
            range(10, 14): 0.75
        }
        
        for spread_range, win_rate in teaser_win_rates.items():
            if int(adjusted_spread) in spread_range:
                return win_rate
        
        return 0.70  # Default conservative estimate
    
    def find_best_teasers(self, tease_points=6, num_legs=2):
        """Find best 2-team or 3-team teasers"""
        
        best_teasers = []
        
        # Get games with spreads that benefit from teasing
        teaser_candidates = self.find_key_numbers()
        
        from itertools import combinations
        
        for combo in combinations(teaser_candidates.itertuples(), num_legs):
            games = []
            total_prob = 1.0
            
            for game in combo:
                # Tease favorites (typically better value)
                is_favorite = game.spread_line < 0
                prob = self.calculate_teaser_probability(game.spread_line, tease_points, is_favorite)
                total_prob *= prob
                
                games.append({
                    'game_id': game.game_id,
                    'original_spread': game.spread_line,
                    'teased_spread': game.spread_line - tease_points if is_favorite else game.spread_line + tease_points,
                    'win_probability': prob
                })
            
            # Standard teaser payouts
            if num_legs == 2:
                teaser_payout = -110  # Bet $110 to win $100
                breakeven = 0.5238
            elif num_legs == 3:
                teaser_payout = 180  # Bet $100 to win $180
                breakeven = 0.357
            
            # Calculate EV
            if teaser_payout < 0:
                ev = (total_prob * (100/abs(teaser_payout))) - (1 - total_prob)
            else:
                ev = (total_prob * (teaser_payout/100)) - (1 - total_prob)
            
            if ev > 0:
                best_teasers.append({
                    'games': games,
                    'combined_probability': total_prob,
                    'expected_value': ev,
                    'roi': ev * 100
                })
        
        return sorted(best_teasers, key=lambda x: x['expected_value'], reverse=True)

# Usage
teaser_opt = TeaserOptimizer(predictions_df)
best_6pt_teasers = teaser_opt.find_best_teasers(tease_points=6, num_legs=2)

print(f"\n📈 Top 5 Best 6-Point Teasers:")
for i, teaser in enumerate(best_6pt_teasers[:5], 1):
    print(f"\n{i}. ROI: {teaser['roi']:.2f}%")
    print(f"   Win Probability: {teaser['combined_probability']*100:.1f}%")
    for game in teaser['games']:
        print(f"   {game['game_id']}: {game['original_spread']} → {game['teased_spread']}")
```

---

## 🔴 Live In-Game Betting

### 29. Real-Time Win Probability Model

**Impact**: VERY HIGH | **Effort**: 20 hours

```python
# Create new file: models/live_win_probability.py

class LiveWinProbability:
    """Calculate real-time win probability during games"""
    
    def __init__(self):
        self.model = None
        
    def create_live_features(self, game_state):
        """
        Extract features from current game state
        
        game_state dict should contain:
        - score_diff: Current score differential (home - away)
        - time_remaining_seconds: Seconds left in game
        - possession: Who has the ball (home/away)
        - field_position: Yards from own goal line
        - down: Current down
        - distance: Yards to go
        - timeouts_home: Home timeouts remaining
        - timeouts_away: Away timeouts remaining
        """
        
        features = {}
        
        # Time-based features
        features['time_remaining_pct'] = game_state['time_remaining_seconds'] / 3600
        features['quarter'] = (3600 - game_state['time_remaining_seconds']) // 900 + 1
        features['is_2min_warning'] = 1 if 100 < game_state['time_remaining_seconds'] < 130 else 0
        
        # Score-based features
        features['score_diff'] = game_state['score_diff']
        features['score_diff_squared'] = game_state['score_diff'] ** 2
        features['abs_score_diff'] = abs(game_state['score_diff'])
        features['is_one_score_game'] = 1 if abs(game_state['score_diff']) <= 8 else 0
        features['is_two_score_game'] = 1 if 8 < abs(game_state['score_diff']) <= 16 else 0
        
        # Possession features
        features['has_possession'] = 1 if game_state['possession'] == 'home' else 0
        features['possession_value'] = features['has_possession'] if game_state['score_diff'] < 0 else -features['has_possession']
        
        # Field position (0-100, 0=own goal, 100=opponent goal)
        features['field_position'] = game_state['field_position']
        features['in_red_zone'] = 1 if game_state['field_position'] >= 80 else 0
        features['in_fg_range'] = 1 if game_state['field_position'] >= 60 else 0
        
        # Situational
        features['down'] = game_state['down']
        features['distance'] = game_state['distance']
        features['is_passing_down'] = 1 if (game_state['down'] == 3 and game_state['distance'] > 5) else 0
        
        # Timeouts
        features['timeout_advantage'] = game_state['timeouts_home'] - game_state['timeouts_away']
        features['has_timeouts'] = game_state['timeouts_home'] + game_state['timeouts_away']
        
        # Interaction terms
        features['score_x_time'] = features['score_diff'] * features['time_remaining_pct']
        features['possession_x_field_pos'] = features['has_possession'] * features['field_position']
        
        return pd.DataFrame([features])
    
    def train_on_historical_pbp(self, play_by_play_df):
        """Train on historical play-by-play data"""
        
        # Filter to key game states (every 2 minutes, key plays)
        key_moments = play_by_play_df[
            (play_by_play_df['game_seconds_remaining'] % 120 == 0) |  # Every 2 minutes
            (play_by_play_df['down'] == 4) |  # All 4th downs
            (play_by_play_df['score_differential'].abs() <= 8)  # Close games
        ].copy()
        
        # Create features for each moment
        X_list = []
        y_list = []
        
        for game_id in key_moments['game_id'].unique():
            game_moments = key_moments[key_moments['game_id'] == game_id].sort_values('game_seconds_remaining', ascending=False)
            
            final_result = game_moments.iloc[-1]['result']  # Home team won = 1
            
            for _, row in game_moments.iterrows():
                game_state = {
                    'score_diff': row['score_differential'],
                    'time_remaining_seconds': row['game_seconds_remaining'],
                    'possession': 'home' if row['posteam'] == row['home_team'] else 'away',
                    'field_position': row['yardline_100'],
                    'down': row['down'],
                    'distance': row['ydstogo'],
                    'timeouts_home': row['home_timeouts_remaining'],
                    'timeouts_away': row['away_timeouts_remaining']
                }
                
                features = self.create_live_features(game_state)
                X_list.append(features)
                y_list.append(final_result)
        
        X = pd.concat(X_list, ignore_index=True)
        y = pd.Series(y_list)
        
        # Train model
        from xgboost import XGBClassifier
        
        self.model = XGBClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8
        )
        
        self.model.fit(X, y)
        
        return self.model
    
    def predict_live_win_prob(self, game_state):
        """Predict win probability for current game state"""
        features = self.create_live_features(game_state)
        win_prob = self.model.predict_proba(features)[0, 1]
        
        return win_prob
    
    def identify_betting_opportunities(self, game_state, market_prob):
        """
        Find +EV live betting opportunities
        
        Args:
            game_state: Current game state
            market_prob: Implied probability from live betting odds
        """
        
        our_prob = self.predict_live_win_prob(game_state)
        edge = our_prob - market_prob
        
        if edge > 0.05:  # 5% edge threshold
            return {
                'bet': 'home' if our_prob > market_prob else 'away',
                'our_probability': our_prob,
                'market_probability': market_prob,
                'edge': edge,
                'kelly_bet_size': edge / (1/market_prob - 1)  # Kelly Criterion
            }
        
        return None

# Usage - integrate with live data feed
live_model = LiveWinProbability()
live_model.train_on_historical_pbp(pbp_data)

# Example: 4th quarter, home team down 3, has ball at midfield
current_state = {
    'score_diff': -3,
    'time_remaining_seconds': 420,  # 7 minutes left
    'possession': 'home',
    'field_position': 50,
    'down': 1,
    'distance': 10,
    'timeouts_home': 2,
    'timeouts_away': 3
}

win_prob = live_model.predict_live_win_prob(current_state)
print(f"Home team win probability: {win_prob*100:.1f}%")

# Check if live odds offer value
market_odds = -130  # Home team live odds
market_prob = abs(market_odds) / (abs(market_odds) + 100)
opportunity = live_model.identify_betting_opportunities(current_state, market_prob)

if opportunity:
    print(f"\n🚨 LIVE BETTING ALERT!")
    print(f"   Bet: {opportunity['bet']}")
    print(f"   Edge: {opportunity['edge']*100:.1f}%")
    print(f"   Kelly bet: {opportunity['kelly_bet_size']*100:.1f}% of bankroll")
```

---

## 📊 Continue to Data Pipeline

See [DATA_PIPELINE_ROADMAP.md](DATA_PIPELINE_ROADMAP.md) for automation and infrastructure improvements.
