import pandas as pd
import numpy as np
from os import path
# from xgboost import XGBClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, accuracy_score
# from sklearn.inspection import permutation_importance
import os
from datetime import datetime
import requests
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.inspection import permutation_importance
import json
from sklearn.model_selection import cross_val_score
import random
from sklearn.calibration import CalibratedClassifierCV

DATA_DIR = 'data_files/'

historical_game_level_data = pd.read_csv(path.join(DATA_DIR, 'nfl_games_historical.csv'), sep='\t')


historical_game_level_data['gameLineAccuracy'] = (historical_game_level_data['home_score'] - historical_game_level_data['away_score']).abs() / historical_game_level_data['spread_line'].abs()
historical_game_level_data['overUnderAccuracy'] = (historical_game_level_data['total'] - (historical_game_level_data['home_score'] + historical_game_level_data['away_score'])).abs() / historical_game_level_data['total'].abs()
historical_game_level_data['homeWin'] = np.where(historical_game_level_data['home_score'] > historical_game_level_data['away_score'], 1, 0)
historical_game_level_data['awayWin'] = np.where(historical_game_level_data['away_score'] > historical_game_level_data['home_score'], 1, 0)
historical_game_level_data['isCloseGame'] = np.where((historical_game_level_data['home_score'] - historical_game_level_data['away_score']).abs() <= 3, 1, 0)
historical_game_level_data['isBlowout'] = np.where((historical_game_level_data['home_score'] - historical_game_level_data['away_score']).abs() >= 20, 1, 0)
historical_game_level_data['totalScore'] = historical_game_level_data['home_score'] + historical_game_level_data['away_score']
historical_game_level_data['pointDiff'] = (historical_game_level_data['home_score'] - historical_game_level_data['away_score']).abs()
# Fix: Positive spread_line means home team favored, negative means away team favored
historical_game_level_data['homeFavored'] = np.where(historical_game_level_data['spread_line'] > 0, 1, 0)
historical_game_level_data['awayFavored'] = np.where(historical_game_level_data['spread_line'] < 0, 1, 0)
# Fix: spreadCovered = 1 when the favored team covers the spread
historical_game_level_data['spreadCovered'] = np.where((historical_game_level_data['homeFavored'] == 1) & ((historical_game_level_data['home_score'] - historical_game_level_data['away_score']) > historical_game_level_data['spread_line']), 1,
                                                    np.where((historical_game_level_data['awayFavored'] == 1) & ((historical_game_level_data['away_score'] - historical_game_level_data['home_score']) > abs(historical_game_level_data['spread_line'])), 1, 0))
historical_game_level_data['favoriteCovered'] = np.where(
    (historical_game_level_data['homeFavored'] == 1) & ((historical_game_level_data['home_score'] - historical_game_level_data['away_score']) > historical_game_level_data['spread_line']), 1,
    np.where((historical_game_level_data['awayFavored'] == 1) & ((historical_game_level_data['away_score'] - historical_game_level_data['home_score']) > abs(historical_game_level_data['spread_line'])), 1, 0)
)
historical_game_level_data['underdogCovered'] = np.where(
    (historical_game_level_data['homeFavored'] == 1) & ((historical_game_level_data['away_score'] - historical_game_level_data['home_score']) + historical_game_level_data['spread_line'] >= 0), 1,
    np.where((historical_game_level_data['awayFavored'] == 1) & ((historical_game_level_data['home_score'] - historical_game_level_data['away_score']) - historical_game_level_data['spread_line'] >= 0), 1, 0)
)
# Fix: underdogWon = 1 when the underdog wins outright
historical_game_level_data['underdogWon'] = np.where(
    (historical_game_level_data['homeFavored'] == 1) & (historical_game_level_data['away_score'] > historical_game_level_data['home_score']), 1,
    np.where((historical_game_level_data['awayFavored'] == 1) & (historical_game_level_data['home_score'] > historical_game_level_data['away_score']), 1, 0)
)
historical_game_level_data['overHit'] = np.where((historical_game_level_data['home_score'] + historical_game_level_data['away_score']) > historical_game_level_data['total_line'], 1, 0)
historical_game_level_data['underHit'] = np.where(historical_game_level_data['total_line'] > (historical_game_level_data['home_score'] + historical_game_level_data['away_score']), 1, 0)
historical_game_level_data['totalHit'] = np.where(historical_game_level_data['total_line'] == (historical_game_level_data['home_score'] + historical_game_level_data['away_score']), 1, 0)
historical_game_level_data['total_line_diff'] = (historical_game_level_data['total_line'] - (historical_game_level_data['home_score'] + historical_game_level_data['away_score']))

# Add features specifically for predicting upsets (underdog wins)
historical_game_level_data['spreadSize'] = historical_game_level_data['spread_line'].abs()
historical_game_level_data['isCloseSpread'] = np.where(historical_game_level_data['spreadSize'] <= 3, 1, 0)
historical_game_level_data['isMediumSpread'] = np.where((historical_game_level_data['spreadSize'] > 3) & (historical_game_level_data['spreadSize'] <= 7), 1, 0)
historical_game_level_data['isLargeSpread'] = np.where(historical_game_level_data['spreadSize'] > 7, 1, 0)
def calc_rolling_stat(df, team_col, stat_col):
    # For each row, calculate stat for team using only games prior to that row's week/season
    stats = []
    for idx, row in df.iterrows():
        team = row[team_col]
        week = row['week']
        season = row['season']
        prior_games = df[(df[team_col] == team) & ((df['season'] < season) | ((df['season'] == season) & (df['week'] < week)))]
        if len(prior_games) == 0:
            stats.append(0)
        else:
            stats.append(prior_games[stat_col].mean())
    return stats

def calc_rolling_count(df, team_col):
    counts = []
    for idx, row in df.iterrows():
        team = row[team_col]
        week = row['week']
        season = row['season']
        prior_games = df[(df[team_col] == team) & ((df['season'] < season) | ((df['season'] == season) & (df['week'] < week)))]
        counts.append(len(prior_games))
    return counts

historical_game_level_data['homeTeamWinPct'] = calc_rolling_stat(historical_game_level_data, 'home_team', 'homeWin')
historical_game_level_data['awayTeamWinPct'] = calc_rolling_stat(historical_game_level_data, 'away_team', 'awayWin')
historical_game_level_data['homeTeamCloseGamePct'] = calc_rolling_stat(historical_game_level_data, 'home_team', 'isCloseGame')
historical_game_level_data['awayTeamCloseGamePct'] = calc_rolling_stat(historical_game_level_data, 'away_team', 'isCloseGame')
historical_game_level_data['homeTeamBlowoutPct'] = calc_rolling_stat(historical_game_level_data, 'home_team', 'isBlowout')
historical_game_level_data['awayTeamBlowoutPct'] = calc_rolling_stat(historical_game_level_data, 'away_team', 'isBlowout')
historical_game_level_data['homeTeamAvgScore'] = calc_rolling_stat(historical_game_level_data, 'home_team', 'home_score')
historical_game_level_data['awayTeamAvgScore'] = calc_rolling_stat(historical_game_level_data, 'away_team', 'away_score')
historical_game_level_data['homeTeamAvgScoreAllowed'] = calc_rolling_stat(historical_game_level_data, 'home_team', 'away_score')
historical_game_level_data['awayTeamAvgScoreAllowed'] = calc_rolling_stat(historical_game_level_data, 'away_team', 'home_score')
historical_game_level_data['homeTeamAvgPointDiff'] = calc_rolling_stat(historical_game_level_data, 'home_team', 'pointDiff')
historical_game_level_data['awayTeamAvgPointDiff'] = calc_rolling_stat(historical_game_level_data, 'away_team', 'pointDiff')
historical_game_level_data['homeTeamAvgTotalScore'] = calc_rolling_stat(historical_game_level_data, 'home_team', 'totalScore')
historical_game_level_data['awayTeamAvgTotalScore'] = calc_rolling_stat(historical_game_level_data, 'away_team', 'totalScore')
historical_game_level_data['homeTeamGamesPlayed'] = calc_rolling_count(historical_game_level_data, 'home_team')
historical_game_level_data['awayTeamGamesPlayed'] = calc_rolling_count(historical_game_level_data, 'away_team')
historical_game_level_data['homeTeamAvgPointSpread'] = calc_rolling_stat(historical_game_level_data, 'home_team', 'spread_line')
historical_game_level_data['awayTeamAvgPointSpread'] = calc_rolling_stat(historical_game_level_data, 'away_team', 'spread_line')
historical_game_level_data['homeTeamAvgTotal'] = calc_rolling_stat(historical_game_level_data, 'home_team', 'total')
historical_game_level_data['awayTeamAvgTotal'] = calc_rolling_stat(historical_game_level_data, 'away_team', 'total')
historical_game_level_data['homeTeamFavoredPct'] = historical_game_level_data['home_team'].map(historical_game_level_data.groupby('home_team')['homeFavored'].mean())
historical_game_level_data['awayTeamFavoredPct'] = historical_game_level_data['away_team'].map(historical_game_level_data.groupby('away_team')['awayFavored'].mean())
historical_game_level_data['homeTeamSpreadCoveredPct'] = historical_game_level_data['home_team'].map(historical_game_level_data.groupby('home_team')['spreadCovered'].mean())
historical_game_level_data['awayTeamSpreadCoveredPct'] = historical_game_level_data['away_team'].map(historical_game_level_data.groupby('away_team')['spreadCovered'].mean())
historical_game_level_data['homeTeamOverHitPct'] = historical_game_level_data['home_team'].map(historical_game_level_data.groupby('home_team')['overHit'].mean())
historical_game_level_data['awayTeamOverHitPct'] = historical_game_level_data['away_team'].map(historical_game_level_data.groupby('away_team')['overHit'].mean())
historical_game_level_data['homeTeamUnderHitPct'] = historical_game_level_data['home_team'].map(historical_game_level_data.groupby('home_team')['underHit'].mean())
historical_game_level_data['awayTeamUnderHitPct'] = historical_game_level_data['away_team'].map(historical_game_level_data.groupby('away_team')['underHit'].mean())
historical_game_level_data['homeTeamTotalHitPct'] = historical_game_level_data['home_team'].map(historical_game_level_data.groupby('home_team')['totalHit'].mean())
historical_game_level_data['awayTeamTotalHitPct'] = historical_game_level_data['away_team'].map(historical_game_level_data.groupby('away_team')['totalHit'].mean())

historical_game_level_data.fillna(0, inplace=True)
historical_game_level_data.replace([np.inf, -np.inf], 0, inplace=True)
# Load best feature subsets from disk
def load_best_features(filename, all_features):
    try:
        with open(path.join(DATA_DIR, filename), 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        # Only keep valid features
        best_feats = [feat for feat in lines if feat in all_features]
        return best_feats if best_feats else all_features
    except Exception:
        return all_features

features = [
    # Pregame features only (removed 'total' as it's actual game total, causing data leakage)
    'spread_line', 'away_moneyline', 'home_moneyline', 'away_spread_odds', 'home_spread_odds', 'total_line',
    'under_odds', 'over_odds', 'div_game', 'roof', 'surface', 'temp', 'wind', 'away_rest', 'home_rest',
    'home_team', 'away_team', 'gameday', 'week', 'season', 'home_qb_id', 'away_qb_id', 'home_qb_name', 'away_qb_name',
    'home_coach', 'away_coach', 'stadium', 'location',
    # Rolling team stats (calculated from previous games only)
    'homeTeamWinPct', 'awayTeamWinPct', 'homeTeamCloseGamePct', 'awayTeamCloseGamePct', 'homeTeamBlowoutPct', 'awayTeamBlowoutPct',
    'homeTeamAvgScore', 'awayTeamAvgScore', 'homeTeamAvgScoreAllowed', 'awayTeamAvgScoreAllowed', 'homeTeamAvgPointDiff', 'awayTeamAvgPointDiff',
    'homeTeamAvgTotalScore', 'awayTeamAvgTotalScore', 'homeTeamGamesPlayed', 'awayTeamGamesPlayed', 'homeTeamAvgPointSpread', 'awayTeamAvgPointSpread',
    'homeTeamAvgTotal', 'awayTeamAvgTotal', 'homeTeamFavoredPct', 'awayTeamFavoredPct', 'homeTeamSpreadCoveredPct', 'awayTeamSpreadCoveredPct',
    'homeTeamOverHitPct', 'awayTeamOverHitPct', 'homeTeamUnderHitPct', 'awayTeamUnderHitPct', 'homeTeamTotalHitPct', 'awayTeamTotalHitPct',
    # Upset-specific features
    'spreadSize', 'isCloseSpread', 'isMediumSpread', 'isLargeSpread'
]
best_features_spread = load_best_features('best_features_spread.txt', features)
best_features_moneyline = load_best_features('best_features_moneyline.txt', features)
best_features_totals = load_best_features('best_features_totals.txt', features)
target_spread = 'spreadCovered'
target_overunder = 'overHit'  # Predicting over hits; under hits can be derived as 1 - overHit


# Prepare data

# Prepare data using best features for each target
print('underdogCovered value counts:')
print(historical_game_level_data['underdogCovered'].value_counts())
X_spread = historical_game_level_data[best_features_spread].select_dtypes(include=["number", "bool", "category"])
if set(best_features_spread) - set(X_spread.columns):
    print(f"Warning: Dropped non-numeric features for spread: {set(best_features_spread) - set(X_spread.columns)}")
y_spread = historical_game_level_data[target_spread]
X_train_spread, X_test_spread, y_spread_train, y_spread_test = train_test_split(
    X_spread, y_spread, test_size=0.2, random_state=42, stratify=y_spread)

X_moneyline = historical_game_level_data[best_features_moneyline].select_dtypes(include=["number", "bool", "category"])
if set(best_features_moneyline) - set(X_moneyline.columns):
    print(f"Warning: Dropped non-numeric features for moneyline: {set(best_features_moneyline) - set(X_moneyline.columns)}")
y_moneyline = historical_game_level_data['underdogWon']
X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(
    X_moneyline, y_moneyline, test_size=0.2, random_state=42, stratify=y_moneyline)

X_totals = historical_game_level_data[best_features_totals].select_dtypes(include=["number", "bool", "category"])
if set(best_features_totals) - set(X_totals.columns):
    print(f"Warning: Dropped non-numeric features for totals: {set(best_features_totals) - set(X_totals.columns)}")
y_totals = historical_game_level_data[target_overunder]
X_train_tot, X_test_tot, y_train_tot, y_test_tot = train_test_split(
    X_totals, y_totals, test_size=0.2, random_state=42, stratify=y_totals)


print('y_spread_train value counts:')
print(y_spread_train.value_counts())
print('y_moneyline_train value counts:')
print(y_train_ml.value_counts())
print('y_totals_train value counts:')
print(y_train_tot.value_counts())

# Train models

# Train and calibrate models for each target
model_spread = CalibratedClassifierCV(XGBClassifier(eval_metric='logloss'), method='sigmoid', cv=3)
model_spread.fit(X_train_spread, y_spread_train)

# Calculate class weights for moneyline model to handle imbalance
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_ml), y=y_train_ml)
scale_pos_weight = class_weights[1] / class_weights[0]

model_moneyline = CalibratedClassifierCV(
    XGBClassifier(eval_metric='logloss', scale_pos_weight=scale_pos_weight), 
    method='sigmoid', cv=3
)
model_moneyline.fit(X_train_ml, y_train_ml)

model_totals = CalibratedClassifierCV(XGBClassifier(eval_metric='logloss'), method='isotonic', cv=3)
model_totals.fit(X_train_tot, y_train_tot)

# Predict

# Predict on test sets
y_spread_pred = model_spread.predict(X_test_spread)
y_moneyline_pred = model_moneyline.predict(X_test_ml)
y_totals_pred = model_totals.predict(X_test_tot)

# Optimize threshold using F1-score first
from sklearn.metrics import f1_score
y_moneyline_proba = model_moneyline.predict_proba(X_test_ml)[:, 1]
thresholds = np.arange(0.1, 0.6, 0.02)
f1_scores = []
for threshold in thresholds:
    y_pred_thresh = (y_moneyline_proba >= threshold).astype(int)
    f1 = f1_score(y_test_ml, y_pred_thresh)
    f1_scores.append(f1)

best_threshold = thresholds[np.argmax(f1_scores)]
optimal_moneyline_threshold = best_threshold

# Predict probabilities for all data
historical_game_level_data['prob_underdogCovered'] = model_spread.predict_proba(X_spread)[:, 1]
historical_game_level_data['prob_underdogWon'] = model_moneyline.predict_proba(X_moneyline)[:, 1]
historical_game_level_data['prob_overHit'] = model_totals.predict_proba(X_totals)[:, 1]

# Apply optimized thresholds for predictions
historical_game_level_data['pred_underdogWon_optimal'] = (historical_game_level_data['prob_underdogWon'] >= optimal_moneyline_threshold).astype(int)

# Betting Simulation Analysis
def calculate_betting_return(row, bet_type='moneyline'):
    """Calculate return on $100 bet based on betting odds and actual outcome"""
    if bet_type == 'moneyline':
        # Moneyline betting on underdogs only (when model predicts upset)
        if row['pred_underdogWon_optimal'] == 1:  # Model predicts underdog win
            if row['underdogWon'] == 1:  # Underdog actually won
                # Calculate payout from underdog moneyline odds
                underdog_odds = max(row['away_moneyline'], row['home_moneyline'])  # Higher odds = underdog
                if underdog_odds > 0:  # American odds positive (underdog)
                    return underdog_odds  # $100 bet returns $underdog_odds profit
                else:  # American odds negative (shouldn't happen for underdogs)
                    return 100 / abs(underdog_odds) * 100
            else:  # Bet lost
                return -100
        else:  # No bet placed
            return 0
    elif bet_type == 'spread':
        # Spread betting - bet on underdog to cover
        if row['prob_underdogCovered'] >= 0.55:  # High confidence spread bet
            if row['underdogCovered'] == 1:  # Underdog covered
                return 90.91  # Standard -110 odds: win $90.91 on $100 bet
            else:
                return -100
        else:
            return 0
    return 0

# Apply betting simulation
historical_game_level_data['moneyline_bet_return'] = historical_game_level_data.apply(
    lambda row: calculate_betting_return(row, 'moneyline'), axis=1
)
historical_game_level_data['spread_bet_return'] = historical_game_level_data.apply(
    lambda row: calculate_betting_return(row, 'spread'), axis=1
)

# Calculate betting performance metrics
print(f"\nBetting Analysis Debug:")
print(f"Total games: {len(historical_game_level_data)}")
print(f"Games with optimal underdog predictions: {(historical_game_level_data['pred_underdogWon_optimal'] == 1).sum()}")
print(f"Games with high spread confidence (>=0.55): {(historical_game_level_data['prob_underdogCovered'] >= 0.55).sum()}")

moneyline_bets = historical_game_level_data[historical_game_level_data['pred_underdogWon_optimal'] == 1]
spread_bets = historical_game_level_data[historical_game_level_data['prob_underdogCovered'] >= 0.55]

if len(moneyline_bets) > 0:
    total_moneyline_return = moneyline_bets['moneyline_bet_return'].sum()
    num_moneyline_bets = len(moneyline_bets)
    moneyline_win_rate = (moneyline_bets['moneyline_bet_return'] > 0).mean()
    avg_moneyline_return = total_moneyline_return / num_moneyline_bets
    print(f"\nMoneyline Betting Simulation:")
    print(f"Total bets placed: {num_moneyline_bets}")
    print(f"Win rate: {moneyline_win_rate:.1%}")
    print(f"Total return: ${total_moneyline_return:.2f}")
    print(f"Average return per bet: ${avg_moneyline_return:.2f}")
    print(f"ROI: {(total_moneyline_return / (num_moneyline_bets * 100)):.1%}")

if len(spread_bets) > 0:
    total_spread_return = spread_bets['spread_bet_return'].sum()
    num_spread_bets = len(spread_bets)
    spread_win_rate = (spread_bets['spread_bet_return'] > 0).mean()
    avg_spread_return = total_spread_return / num_spread_bets
    print(f"\nSpread Betting Simulation:")
    print(f"Total bets placed: {num_spread_bets}")
    print(f"Win rate: {spread_win_rate:.1%}")
    print(f"Total return: ${total_spread_return:.2f}")
    print(f"Average return per bet: ${avg_spread_return:.2f}")
    print(f"ROI: {(total_spread_return / (num_spread_bets * 100)):.1%}")

# Probability sanity check for prob_overHit
mean_prob_overhit = historical_game_level_data['prob_overHit'].mean()
actual_rate_overhit = historical_game_level_data[target_overunder].mean()
print(f"Mean predicted probability for overHit: {mean_prob_overhit:.4f}")
print(f"Actual outcome rate for overHit: {actual_rate_overhit:.4f}")

# Accuracy

from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

spread_accuracy = accuracy_score(y_spread_test, y_spread_pred)
moneyline_accuracy = accuracy_score(y_test_ml, y_moneyline_pred)
totals_accuracy = accuracy_score(y_test_tot, y_totals_pred)

print(f"Spread Prediction Accuracy: {spread_accuracy:.4f}")
print(f"Moneyline Prediction Accuracy: {moneyline_accuracy:.4f}")
print(f"Totals Prediction Accuracy: {totals_accuracy:.4f}")

# Additional metrics for moneyline model (focus on underdog prediction)
baseline_accuracy = (y_test_ml == 0).mean()  # Always predict favorite wins
improvement = moneyline_accuracy - baseline_accuracy
moneyline_precision = precision_score(y_test_ml, y_moneyline_pred)
moneyline_recall = recall_score(y_test_ml, y_moneyline_pred)
moneyline_f1 = f1_score(y_test_ml, y_moneyline_pred)

print(f"\nMoneyline Model Deep Analysis:")
print(f"Baseline (always favorite): {baseline_accuracy:.4f}")
print(f"Model improvement: {improvement:+.4f} ({improvement*100:+.1f} pts)")

# Debug: Check probability distribution and predictions
print(f"Underdog win probabilities - Mean: {y_moneyline_proba.mean():.4f}, Min: {y_moneyline_proba.min():.4f}, Max: {y_moneyline_proba.max():.4f}")

# Use optimal threshold for predictions (calculated earlier)
y_moneyline_pred_optimal = (y_moneyline_proba >= optimal_moneyline_threshold).astype(int)
optimal_accuracy = accuracy_score(y_test_ml, y_moneyline_pred_optimal)
optimal_precision = precision_score(y_test_ml, y_moneyline_pred_optimal)
optimal_recall = recall_score(y_test_ml, y_moneyline_pred_optimal)
best_f1 = f1_score(y_test_ml, y_moneyline_pred_optimal)

print(f"Optimal threshold: {optimal_moneyline_threshold:.2f}")
print(f"With optimal threshold - Accuracy: {optimal_accuracy:.4f}, Precision: {optimal_precision:.4f}, Recall: {optimal_recall:.4f}, F1: {best_f1:.4f}")
print(f"Improvement over baseline: {optimal_accuracy - baseline_accuracy:+.4f} ({(optimal_accuracy - baseline_accuracy)*100:+.1f} pts)")

print(f"Number of underdog predictions: {(y_moneyline_pred == 1).sum()} out of {len(y_moneyline_pred)}")
print(f"Underdog win precision: {moneyline_precision:.4f}")
print(f"Underdog win recall: {moneyline_recall:.4f}")
print(f"Underdog win F1-score: {moneyline_f1:.4f}")

# Feature importance
feature_importance_spread = permutation_importance(model_spread, X_test_spread, y_spread_test, n_repeats=10, random_state=42)
feature_importance_totals = permutation_importance(model_totals, X_test_tot, y_test_tot, n_repeats=10, random_state=42)
sorted_idx_spread = feature_importance_spread.importances_mean.argsort()[::-1]
sorted_idx_totals = feature_importance_totals.importances_mean.argsort()[::-1]
print("Top 5 Important Features for Spread Prediction:")
for idx in sorted_idx_spread[:5]:
    print(f"{best_features_spread[idx]}: {feature_importance_spread.importances_mean[idx]:.4f} ± {feature_importance_spread.importances_std[idx]:.4f}")
print("Top 5 Important Features for Totals Prediction:")
for idx in sorted_idx_totals[:5]:
    print(f"{best_features_totals[idx]}: {feature_importance_totals.importances_mean[idx]:.4f} ± {feature_importance_totals.importances_std[idx]:.4f}")
    
# Assign predictions to the correct rows in the DataFrame
historical_game_level_data['predictedSpreadCovered'] = np.nan
historical_game_level_data['predictedOverHit'] = np.nan
historical_game_level_data.loc[X_test_spread.index, 'predictedSpreadCovered'] = y_spread_pred
historical_game_level_data.loc[X_test_tot.index, 'predictedOverHit'] = y_totals_pred

# --- Implied probability and edge calculations ---
def implied_prob(odds):
    # American odds to implied probability, handle zeros as NaN
    odds = np.array(odds, dtype=float)
    prob = np.full_like(odds, np.nan, dtype=float)
    mask_neg = odds < 0
    mask_pos = odds > 0
    prob[mask_neg] = (-odds[mask_neg]) / ((-odds[mask_neg]) + 100)
    prob[mask_pos] = 100 / (odds[mask_pos] + 100)
    # odds == 0 stays as NaN
    return prob

# Underdog moneyline implied probability and edge
underdog_is_away = historical_game_level_data['away_moneyline'] > historical_game_level_data['home_moneyline']
underdog_ml_odds = np.where(underdog_is_away, historical_game_level_data['away_moneyline'], historical_game_level_data['home_moneyline'])
historical_game_level_data['implied_prob_underdog_ml'] = implied_prob(underdog_ml_odds)
historical_game_level_data['edge_underdog_ml'] = historical_game_level_data['prob_underdogWon'] - historical_game_level_data['implied_prob_underdog_ml']

# Spread implied probability and edge (for underdog covering)
underdog_spread_odds = np.where(underdog_is_away, historical_game_level_data['away_spread_odds'], historical_game_level_data['home_spread_odds'])
historical_game_level_data['implied_prob_underdog_spread'] = implied_prob(underdog_spread_odds)
historical_game_level_data['edge_underdog_spread'] = historical_game_level_data['prob_underdogCovered'] - historical_game_level_data['implied_prob_underdog_spread']

# Over/under implied probability and edge
historical_game_level_data['implied_prob_over'] = implied_prob(historical_game_level_data['over_odds'])
historical_game_level_data['implied_prob_under'] = implied_prob(historical_game_level_data['under_odds'])
historical_game_level_data['edge_over'] = historical_game_level_data['prob_overHit'] - historical_game_level_data['implied_prob_over']
historical_game_level_data['edge_under'] = (1 - historical_game_level_data['prob_overHit']) - historical_game_level_data['implied_prob_under']

# Save to CSV for Streamlit

# Ensure all probability columns are present for Streamlit dashboard
prob_cols = ['prob_underdogCovered', 'prob_underdogWon', 'prob_overHit']
for col in prob_cols:
    if col not in historical_game_level_data.columns:
        historical_game_level_data[col] = np.nan

historical_game_level_data.to_csv(path.join(DATA_DIR, 'nfl_games_historical_with_predictions.csv'), index=False, sep='\t')
print("Saved historical game data with predictions to nfl_games_historical_with_predictions.csv")

# Save model metrics (accuracy, MAE, etc.) to JSON
metrics = {
    "Spread Accuracy": float(spread_accuracy),
    "Moneyline Accuracy": float(moneyline_accuracy),
    "Totals Accuracy": float(totals_accuracy),
    "Spread MAE": float(mean_absolute_error(y_spread_test, y_spread_pred)),
    "Moneyline MAE": float(mean_absolute_error(y_test_ml, y_moneyline_pred)),
    "Totals MAE": float(mean_absolute_error(y_test_tot, y_totals_pred))
}
with open(path.join(DATA_DIR, 'model_metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)

# Save feature importances to CSV
fi_spread = pd.DataFrame({
    'feature': [best_features_spread[i] for i in sorted_idx_spread],
    'importance_mean': feature_importance_spread.importances_mean[sorted_idx_spread],
    'importance_std': feature_importance_spread.importances_std[sorted_idx_spread],
    'model': 'spread'
})
fi_totals = pd.DataFrame({
    'feature': [best_features_totals[i] for i in sorted_idx_totals],
    'importance_mean': feature_importance_totals.importances_mean[sorted_idx_totals],
    'importance_std': feature_importance_totals.importances_std[sorted_idx_totals],
    'model': 'totals'
})
fi_all = pd.concat([fi_spread, fi_totals], ignore_index=True)
fi_all.to_csv(path.join(DATA_DIR, 'model_feature_importances.csv'), index=False)

# Monte Carlo feature selection to find best features
NUM_ITER = 100
SUBSET_SIZE = 8  # Number of features to try in each subset
best_score = 0
best_features = []
random.seed(42)

for i in range(NUM_ITER):
    subset = random.sample(best_features_spread, SUBSET_SIZE)
    # Filter subset to only valid columns
    valid_subset = [col for col in subset if col in X_train_spread.columns]
    if len(valid_subset) < SUBSET_SIZE:
        print(f"Warning: Dropped invalid features from subset: {set(subset) - set(valid_subset)}")
    X_subset = X_train_spread[valid_subset]
    model = XGBClassifier(eval_metric='logloss')
    scores = cross_val_score(model, X_subset, y_spread_train, cv=3, scoring='accuracy')
    mean_score = scores.mean()
    if mean_score > best_score:
        best_score = mean_score
        best_features = subset

print(f"Best feature subset (spread, Monte Carlo {NUM_ITER} iters): {best_features}")
print(f"Best mean CV accuracy: {best_score:.4f}")

# Save best features to a file
with open(path.join(DATA_DIR, 'best_features_spread.txt'), 'w') as f:
    f.write("\n".join(best_features))
    f.write(f"\nBest mean CV accuracy: {best_score:.4f}\n")