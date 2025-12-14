import pandas as pd
from datetime import datetime

# Load predictions
df = pd.read_csv('data_files/nfl_games_historical_with_predictions.csv', sep='\t')
df['gameday'] = pd.to_datetime(df['gameday'], errors='coerce')

# Filter for upcoming games
today = pd.Timestamp.now().normalize()
upcoming = df[df['gameday'] > today]

print("=== UNDERDOG BETTING ANALYSIS ===\n")
print(f'Total upcoming games: {len(upcoming)}')
print(f'\nDistribution of prob_underdogWon:')
print(f'  >= 50%: {(upcoming["prob_underdogWon"] >= 0.50).sum()} games')
print(f'  >= 40%: {(upcoming["prob_underdogWon"] >= 0.40).sum()} games')
print(f'  >= 30%: {(upcoming["prob_underdogWon"] >= 0.30).sum()} games')
print(f'  >= 28%: {(upcoming["prob_underdogWon"] >= 0.28).sum()} games')
print(f'  < 28%: {(upcoming["prob_underdogWon"] < 0.28).sum()} games')

print(f'\nWith positive EV:')
print(f'  prob >= 28% AND ev > 0: {((upcoming["prob_underdogWon"] >= 0.28) & (upcoming["ev_moneyline"] > 0)).sum()} games')

print(f'\nMax prob_underdogWon in upcoming games: {upcoming["prob_underdogWon"].max():.1%}')
print(f'Min prob_underdogWon in upcoming games: {upcoming["prob_underdogWon"].min():.1%}')
print(f'Avg prob_underdogWon in upcoming games: {upcoming["prob_underdogWon"].mean():.1%}')

print(f'\n=== TOP 10 UNDERDOG PROBABILITIES ===')
top_10 = upcoming.nlargest(10, 'prob_underdogWon')[['gameday', 'home_team', 'away_team', 'prob_underdogWon', 'ev_moneyline']]
print(top_10.to_string(index=False))

# Check historical games for comparison
historical = df[df['gameday'] <= today]
print(f'\n=== HISTORICAL COMPARISON (completed games) ===')
print(f'Total historical games: {len(historical)}')
print(f'Historical games with prob_underdogWon >= 28%: {(historical["prob_underdogWon"] >= 0.28).sum()} ({(historical["prob_underdogWon"] >= 0.28).sum() / len(historical) * 100:.1f}%)')
print(f'Max historical prob_underdogWon: {historical["prob_underdogWon"].max():.1%}')
