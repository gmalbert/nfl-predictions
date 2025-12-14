import pandas as pd
from datetime import datetime

# Load predictions
df = pd.read_csv('data_files/nfl_games_historical_with_predictions.csv', sep='\t')
df['gameday'] = pd.to_datetime(df['gameday'], errors='coerce')

# Filter for upcoming games
today = pd.Timestamp.now().normalize()
upcoming = df[df['gameday'] > today]

print(f'Total upcoming games: {len(upcoming)}')
print(f'Games with pred_underdogWon_optimal=1: {(upcoming["pred_underdogWon_optimal"] == 1).sum()}')
print(f'Games with pred_underdogWon_optimal=0: {(upcoming["pred_underdogWon_optimal"] == 0).sum()}')
print(f'\nSample of prob_underdogWon values:')
print(upcoming[['gameday', 'home_team', 'away_team', 'prob_underdogWon', 'pred_underdogWon_optimal']].head(20))

# Check the threshold
print(f'\n\nGames with prob_underdogWon >= 0.28:')
high_prob = upcoming[upcoming['prob_underdogWon'] >= 0.28]
print(f'Count: {len(high_prob)}')
if len(high_prob) > 0:
    print(high_prob[['gameday', 'home_team', 'away_team', 'prob_underdogWon', 'pred_underdogWon_optimal']].head(10))
