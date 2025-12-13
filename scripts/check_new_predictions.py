import pandas as pd
from datetime import datetime

df = pd.read_csv('data_files/nfl_games_historical_with_predictions.csv', sep='\t')
df['gameday'] = pd.to_datetime(df['gameday'])
future = df[df['gameday'] > datetime.now()]

print(f'Total 2025 games: {len(future)}')
print(f'Games >=52.4% spread confidence: {len(future[future["prob_underdogCovered"] >= 0.524])}')
print(f'Games >=45% spread confidence: {len(future[future["prob_underdogCovered"] >= 0.45])}')
print(f'Max spread confidence: {future["prob_underdogCovered"].max():.1%}')
print(f'Games with +EV: {len(future[future["ev_spread"] > 0])}')

if len(future[future['prob_underdogCovered'] >= 0.524]) > 0:
    print('\n+EV Games (>=52.4%):')
    cols = ['away_team', 'home_team', 'prob_underdogCovered', 'ev_spread']
    print(future[future['prob_underdogCovered'] >= 0.524][cols].to_string())
else:
    print('\nNo games above 52.4% threshold yet')
    print('\nTop 5 confidence games:')
    top5 = future.nlargest(5, 'prob_underdogCovered')[['gameday', 'away_team', 'home_team', 'prob_underdogCovered', 'ev_spread']]
    print(top5.to_string())
