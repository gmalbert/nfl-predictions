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
historical_game_level_data['homeFavored'] = np.where(historical_game_level_data['spread_line'] < 0, 1, 0)
historical_game_level_data['awayFavored'] = np.where(historical_game_level_data['spread_line'] > 0, 1, 0)
historical_game_level_data['spreadCovered'] = np.where((historical_game_level_data['homeFavored'] == 1) & ((historical_game_level_data['home_score'] - historical_game_level_data['away_score']) > historical_game_level_data['spread_line']), 1,
                                                    np.where((historical_game_level_data['awayFavored'] == 1) & ((historical_game_level_data['away_score'] - historical_game_level_data['home_score']) > abs(historical_game_level_data['spread_line'])), 1, 0))
historical_game_level_data['overHit'] = np.where((historical_game_level_data['home_score'] + historical_game_level_data['away_score']) > historical_game_level_data['total_line'], 1, 0)
historical_game_level_data['underHit'] = np.where(historical_game_level_data['total_line'] > (historical_game_level_data['home_score'] + historical_game_level_data['away_score']), 1, 0)
historical_game_level_data['totalHit'] = np.where(historical_game_level_data['total_line'] == (historical_game_level_data['home_score'] + historical_game_level_data['away_score']), 1, 0)
historical_game_level_data['total_line_diff'] = (historical_game_level_data['total_line'] - (historical_game_level_data['home_score'] + historical_game_level_data['away_score']))
historical_game_level_data['homeTeamWinPct'] = historical_game_level_data['home_team'].map(historical_game_level_data.groupby('home_team')['homeWin'].mean())
historical_game_level_data['awayTeamWinPct'] = historical_game_level_data['away_team'].map(historical_game_level_data.groupby('away_team')['awayWin'].mean())
historical_game_level_data['homeTeamCloseGamePct'] = historical_game_level_data['home_team'].map(historical_game_level_data.groupby('home_team')['isCloseGame'].mean())
historical_game_level_data['awayTeamCloseGamePct'] = historical_game_level_data['away_team'].map(historical_game_level_data.groupby('away_team')['isCloseGame'].mean())
historical_game_level_data['homeTeamBlowoutPct'] = historical_game_level_data['home_team'].map(historical_game_level_data.groupby('home_team')['isBlowout'].mean())
historical_game_level_data['awayTeamBlowoutPct'] = historical_game_level_data['away_team'].map(historical_game_level_data.groupby('away_team')['isBlowout'].mean())
historical_game_level_data['homeTeamAvgScore'] = historical_game_level_data['home_team'].map(historical_game_level_data.groupby('home_team')['home_score'].mean())
historical_game_level_data['awayTeamAvgScore'] = historical_game_level_data['away_team'].map(historical_game_level_data.groupby('away_team')['away_score'].mean())
historical_game_level_data['homeTeamAvgScoreAllowed'] = historical_game_level_data['home_team'].map(historical_game_level_data.groupby('home_team')['away_score'].mean())
historical_game_level_data['awayTeamAvgScoreAllowed'] = historical_game_level_data['away_team'].map(historical_game_level_data.groupby('away_team')['home_score'].mean())
historical_game_level_data['homeTeamAvgPointDiff'] = historical_game_level_data['home_team'].map(historical_game_level_data.groupby('home_team')['pointDiff'].mean())
historical_game_level_data['awayTeamAvgPointDiff'] = historical_game_level_data['away_team'].map(historical_game_level_data.groupby('away_team')['pointDiff'].mean())
historical_game_level_data['homeTeamAvgTotalScore'] = historical_game_level_data['home_team'].map(historical_game_level_data.groupby('home_team')['totalScore'].mean())
historical_game_level_data['awayTeamAvgTotalScore'] = historical_game_level_data['away_team'].map(historical_game_level_data.groupby('away_team')['totalScore'].mean())
historical_game_level_data['homeTeamGamesPlayed'] = historical_game_level_data['home_team'].map(historical_game_level_data.groupby('home_team')['game_id'].count())
historical_game_level_data['awayTeamGamesPlayed'] = historical_game_level_data['away_team'].map(historical_game_level_data.groupby('away_team')['game_id'].count())
historical_game_level_data['homeTeamAvgPointSpread'] = historical_game_level_data['home_team'].map(historical_game_level_data.groupby('home_team')['spread_line'].mean())
historical_game_level_data['awayTeamAvgPointSpread'] = historical_game_level_data['away_team'].map(historical_game_level_data.groupby('away_team')['spread_line'].mean())
historical_game_level_data['homeTeamAvgTotal'] = historical_game_level_data['home_team'].map(historical_game_level_data.groupby('home_team')['total'].mean())
historical_game_level_data['awayTeamAvgTotal'] = historical_game_level_data['away_team'].map(historical_game_level_data.groupby('away_team')['total'].mean())
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
features = [
    'spread_line', 'total', 'homeTeamWinPct', 'awayTeamWinPct', 'homeTeamCloseGamePct', 'awayTeamCloseGamePct',
    'homeTeamBlowoutPct', 'awayTeamBlowoutPct', 'homeTeamAvgScore', 'awayTeamAvgScore', 'homeTeamAvgScoreAllowed',
    'awayTeamAvgScoreAllowed', 'homeTeamAvgPointDiff', 'awayTeamAvgPointDiff', 'homeTeamAvgTotalScore',
    'awayTeamAvgTotalScore', 'homeTeamGamesPlayed', 'awayTeamGamesPlayed', 'homeTeamAvgPointSpread',
    'awayTeamAvgPointSpread', 'homeTeamAvgTotal', 'awayTeamAvgTotal', 'homeTeamFavoredPct', 'awayTeamFavoredPct',
    'homeTeamSpreadCoveredPct', 'awayTeamSpreadCoveredPct', 'homeTeamOverHitPct', 'awayTeamOverHitPct',
    'homeTeamUnderHitPct', 'awayTeamUnderHitPct', 'homeTeamTotalHitPct', 'awayTeamTotalHitPct', 'total_line_diff'
]
target_spread = 'spreadCovered'
target_overunder = 'overHit'  # Predicting over hits; under hits can be derived as 1 - overHit


# Prepare data
X = historical_game_level_data[features]
y_spread = historical_game_level_data[target_spread]
y_overunder = historical_game_level_data[target_overunder]
# Stratified split for spreadCovered
X_train, X_test, y_spread_train, y_spread_test = train_test_split(
    X, y_spread, test_size=0.2, random_state=42, stratify=y_spread)
# Stratified split for overHit
X_over_train, X_over_test, y_overunder_train, y_overunder_test = train_test_split(
    X, y_overunder, test_size=0.2, random_state=42, stratify=y_overunder)

print('y_spread_train value counts:')
print(y_spread_train.value_counts())
print('y_overunder_train value counts:')
print(y_overunder_train.value_counts())

# Train models
model_spread = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model_overunder = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model_spread.fit(X_train, y_spread_train)
model_overunder.fit(X_train, y_overunder_train)

# Predict
y_spread_pred = model_spread.predict(X_test)
y_overunder_pred = model_overunder.predict(X_test)

# Accuracy
spread_accuracy = accuracy_score(y_spread_test, y_spread_pred)
overunder_accuracy = accuracy_score(y_overunder_test, y_overunder_pred)
print(f"Spread Prediction Accuracy: {spread_accuracy:.4f}")
print(f"Over/Under Prediction Accuracy: {overunder_accuracy:.4f}")

# Feature importance
feature_importance_spread = permutation_importance(model_spread, X_test, y_spread_test, n_repeats=10, random_state=42)
feature_importance_overunder = permutation_importance(model_overunder, X_test, y_overunder_test, n_repeats=10, random_state=42)
sorted_idx_spread = feature_importance_spread.importances_mean.argsort()[::-1]
sorted_idx_overunder = feature_importance_overunder.importances_mean.argsort()[::-1]
print("Top 5 Important Features for Spread Prediction:")
for idx in sorted_idx_spread[:5]:
    print(f"{features[idx]}: {feature_importance_spread.importances_mean[idx]:.4f} ± {feature_importance_spread.importances_std[idx]:.4f}")
print("Top 5 Important Features for Over/Under Prediction:")
for idx in sorted_idx_overunder[:5]:
    print(f"{features[idx]}: {feature_importance_overunder.importances_mean[idx]:.4f} ± {feature_importance_overunder.importances_std[idx]:.4f}")
    
# Assign predictions to the correct rows in the DataFrame
historical_game_level_data['predictedSpreadCovered'] = np.nan
historical_game_level_data['predictedOverHit'] = np.nan
historical_game_level_data.loc[X_test.index, 'predictedSpreadCovered'] = y_spread_pred
historical_game_level_data.loc[X_over_test.index, 'predictedOverHit'] = y_overunder_pred

# Save to CSV for Streamlit
historical_game_level_data.to_csv(path.join(DATA_DIR, 'nfl_games_historical_with_predictions.csv'), index=False, sep='\t')
print("Saved historical game data with predictions to nfl_games_historical_with_predictions.csv")