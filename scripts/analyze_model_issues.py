"""
Analyze model performance issues and propose improvements
"""
import pandas as pd
import numpy as np
from datetime import datetime

df = pd.read_csv('data_files/nfl_games_historical_with_predictions.csv', sep='\t')
df['gameday'] = pd.to_datetime(df['gameday'])

# Split into training period (2020-2024) and current season (2025)
train_data = df[df['season'] < 2025]
current_season = df[df['season'] == 2025]

print("=" * 80)
print("MODEL PERFORMANCE ANALYSIS")
print("=" * 80)

# 1. Calibration Analysis
print("\n1. CALIBRATION QUALITY (How well do predictions match reality?)")
print("-" * 80)

# Spread predictions
if len(train_data[train_data['underdogCovered'].notna()]) > 0:
    train_spread = train_data[train_data['underdogCovered'].notna()]
    
    # Bin predictions by confidence level
    bins = [0, 0.4, 0.45, 0.5, 0.55, 0.6, 1.0]
    labels = ['<40%', '40-45%', '45-50%', '50-55%', '55-60%', '60%+']
    train_spread['confidence_bin'] = pd.cut(train_spread['prob_underdogCovered'], bins=bins, labels=labels)
    
    calibration = train_spread.groupby('confidence_bin').agg({
        'prob_underdogCovered': 'mean',
        'underdogCovered': 'mean',
        'game_id': 'count'
    }).round(3)
    calibration.columns = ['Predicted', 'Actual', 'Count']
    calibration['Error'] = (calibration['Predicted'] - calibration['Actual']).abs()
    
    print("\nSpread Betting Calibration:")
    print(calibration.to_string())
    print(f"\nAverage calibration error: {calibration['Error'].mean():.3f}")

# 2. Feature Importance Issues
print("\n\n2. FEATURE UTILIZATION")
print("-" * 80)
feature_imp = pd.read_csv('data_files/model_feature_importances.csv')
spread_features = feature_imp[feature_imp['model'] == 'spread'].sort_values('importance_mean', ascending=False)

print("\nTop 10 Features for Spread Model:")
print(spread_features.head(10)[['feature', 'importance_mean']].to_string(index=False))

# Check if momentum features are being used
momentum_features = spread_features[spread_features['feature'].str.contains('Last3|Trend|rest|Weather', case=False, na=False)]
print(f"\nMomentum/Rest/Weather features in top 20: {len(momentum_features.head(20))}/20")

# 3. Prediction Distribution
print("\n\n3. PREDICTION CONFIDENCE DISTRIBUTION")
print("-" * 80)

# Current season predictions
if len(current_season) > 0:
    future_games = current_season[current_season['gameday'] > datetime(2025, 12, 13)]
    
    print(f"\nRemaining 2025 games: {len(future_games)}")
    print(f"\nSpread confidence distribution:")
    print(f"  <40%:    {len(future_games[future_games['prob_underdogCovered'] < 0.40])} games")
    print(f"  40-45%:  {len(future_games[(future_games['prob_underdogCovered'] >= 0.40) & (future_games['prob_underdogCovered'] < 0.45)])} games")
    print(f"  45-50%:  {len(future_games[(future_games['prob_underdogCovered'] >= 0.45) & (future_games['prob_underdogCovered'] < 0.50)])} games")
    print(f"  50-52.4%: {len(future_games[(future_games['prob_underdogCovered'] >= 0.50) & (future_games['prob_underdogCovered'] < 0.524)])} games")
    print(f"  52.4%+:  {len(future_games[future_games['prob_underdogCovered'] >= 0.524])} games (PROFITABLE)")
    
    print(f"\nConfidence stats:")
    print(f"  Mean: {future_games['prob_underdogCovered'].mean():.1%}")
    print(f"  Median: {future_games['prob_underdogCovered'].median():.1%}")
    print(f"  Std Dev: {future_games['prob_underdogCovered'].std():.1%}")

# 4. Training Data Analysis
print("\n\n4. TRAINING DATA QUALITY")
print("-" * 80)

print(f"\nTotal training games (2020-2024): {len(train_data)}")
print(f"Games with spread outcomes: {len(train_data[train_data['underdogCovered'].notna()])}")
print(f"\nClass balance (spread):")
if len(train_data[train_data['underdogCovered'].notna()]) > 0:
    balance = train_data['underdogCovered'].value_counts()
    print(f"  Underdog covered: {balance.get(1, 0)} ({balance.get(1, 0)/len(train_data[train_data['underdogCovered'].notna()])*100:.1f}%)")
    print(f"  Favorite covered: {balance.get(0, 0)} ({balance.get(0, 0)/len(train_data[train_data['underdogCovered'].notna()])*100:.1f}%)")

# 5. EV Analysis
print("\n\n5. EXPECTED VALUE REALITY CHECK")
print("-" * 80)

if len(train_data) > 0:
    # Calculate actual ROI if betting on all historical predictions above threshold
    for threshold in [0.45, 0.50, 0.524]:
        bets = train_data[(train_data['prob_underdogCovered'] >= threshold) & (train_data['underdogCovered'].notna())]
        if len(bets) > 0:
            wins = bets['underdogCovered'].sum()
            win_rate = wins / len(bets)
            # Standard -110 odds: win $90.91 per $100 bet
            roi = (wins * 90.91 - (len(bets) - wins) * 100) / (len(bets) * 100) * 100
            
            print(f"\nIf betting on all games ≥{threshold:.1%} confidence:")
            print(f"  Games: {len(bets)}")
            print(f"  Win rate: {win_rate:.1%}")
            print(f"  ROI: {roi:+.1f}%")
            print(f"  Would be: {'✅ PROFITABLE' if roi > 0 else '❌ LOSING'}")

print("\n" + "=" * 80)
print("KEY ISSUES IDENTIFIED:")
print("=" * 80)
print()
