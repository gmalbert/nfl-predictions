import pandas as pd
from datetime import datetime

df = pd.read_csv('data_files/nfl_games_historical_with_predictions.csv', sep='\t')
df_2025 = df[df['season'] == 2025].copy()

print("="*60)
print("2025 SPREAD PREDICTIONS CHECK")
print("="*60)

print(f"\n📅 2025 Games: {len(df_2025)}")
print(f"📊 Games with pred_spreadCovered_optimal == 1: {(df_2025['pred_spreadCovered_optimal'] == 1).sum()}")
print(f"📈 Games with prob_underdogCovered > 0.50: {(df_2025['prob_underdogCovered'] > 0.50).sum()}")

print(f"\n📊 prob_underdogCovered Distribution:")
print(df_2025['prob_underdogCovered'].describe())

# Check for future games
df_2025['gameday'] = pd.to_datetime(df_2025['gameday'], errors='coerce')
today = pd.to_datetime(datetime.now().date())
future_games = df_2025[df_2025['gameday'] > today]

print(f"\n📅 Future 2025 Games (after today):")
print(f"   Total: {len(future_games)}")
print(f"   With spread bets (pred_spreadCovered_optimal == 1): {(future_games['pred_spreadCovered_optimal'] == 1).sum()}")
print(f"   With prob_underdogCovered > 0.50: {(future_games['prob_underdogCovered'] > 0.50).sum()}")

if len(future_games) > 0:
    print(f"\n🔍 Sample Future Games:")
    sample = future_games.head(3)[['gameday', 'away_team', 'home_team', 'prob_underdogCovered', 'pred_spreadCovered_optimal']]
    print(sample.to_string(index=False))

print("="*60)
